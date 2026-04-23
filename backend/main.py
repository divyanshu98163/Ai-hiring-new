from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
load_dotenv()

import httpx
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from ml_service import DEFAULT_KAGGLE_SOURCE, ResumeMLService, extract_skills
from supabase_client import SupabaseError, SupabaseRestClient


app = FastAPI(title="AI Hiring Backend", version="1.0.0")
ml_service = ResumeMLService()

FREE_TRIAL_CREDITS = 100
MAX_FILE_SIZE_BYTES = 8 * 1024 * 1024
MODEL_VERSION = "resume-ml-v1"
SUPPORTED_RESUME_EXTENSIONS = {".pdf", ".docx", ".txt"}

ACTION_COSTS = {
    "candidate_refresh": 4,
    "job_publish": 25,
    "job_refresh": 5,
    "recruiter_refresh": 8,
    "resume_scan": 15,
}

PRICING_PLANS = [
    {
        "id": "trial",
        "name": "Free Trial",
        "headline": "100 demo credits",
        "description": "Test the resume analyzer and AI matching before you pay.",
        "amount": 0,
        "credits": 100,
        "currency": "INR",
        "cta": "Activate trial",
        "features": [
            "100 one-time demo credits",
            "Resume analysis and ML scoring",
            "Candidate and recruiter dashboards",
        ],
        "recommended": False,
    },
    {
        "id": "growth_500",
        "name": "Growth",
        "headline": "500 credits",
        "description": "For small hiring teams running weekly resume reviews.",
        "amount": 99900,
        "credits": 500,
        "currency": "INR",
        "cta": "Buy 500 credits",
        "features": [
            "500 platform credits",
            "Razorpay checkout",
            "Priority email support",
        ],
        "recommended": True,
    },
    {
        "id": "scale_1500",
        "name": "Scale",
        "headline": "1500 credits",
        "description": "For active recruiters posting roles and refreshing shortlists daily.",
        "amount": 249900,
        "credits": 1500,
        "currency": "INR",
        "cta": "Buy 1500 credits",
        "features": [
            "1500 platform credits",
            "Lower cost per analysis",
            "Best for recruiter teams",
        ],
        "recommended": False,
    },
]


@dataclass
class AuthContext:
    access_token: str
    client: SupabaseRestClient
    role: str
    user: dict[str, Any]


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    if isinstance(exc.detail, dict):
        payload = exc.detail
    else:
        payload = {"error": str(exc.detail)}
    return JSONResponse(payload, status_code=exc.status_code)


@app.exception_handler(SupabaseError)
async def supabase_exception_handler(_: Request, exc: SupabaseError) -> JSONResponse:
    print(f"SupabaseError: {exc}")
    return JSONResponse({"error": str(exc)}, status_code=400)


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        {"error": str(exc) if isinstance(exc, Exception) else "Unexpected backend error."},
        status_code=500,
    )


@app.get("/health")
async def health() -> dict[str, Any]:
    metadata = {}
    try:
        metadata = ml_service.export_metadata()
    except Exception:
        metadata = {
            "dataset": DEFAULT_KAGGLE_SOURCE,
            "model_path": str(Path("backend/models/resume_model.joblib")),
        }

    return {
        "actionCosts": ACTION_COSTS,
        "model": metadata,
        "status": "ok",
    }


@app.get("/jobs")
async def public_jobs() -> dict[str, Any]:
    client = SupabaseRestClient()
    jobs = await get_public_jobs(client)
    return {"jobs": jobs}


@app.get("/billing/plans")
async def billing_plans(request: Request) -> dict[str, Any]:
    auth = await get_auth_context(request, required=False)
    credits = None
    if auth:
        credits = await get_credit_summary(auth.client, auth.user["id"])

    return {
        "actionCosts": ACTION_COSTS,
        "credits": credits,
        "ml": ml_service.export_metadata(),
        "plans": [serialize_plan(plan) for plan in PRICING_PLANS],
        "razorpayEnabled": bool(
            os.environ.get("RAZORPAY_KEY_ID") and os.environ.get("NEXT_PUBLIC_RAZORPAY_KEY_ID")
        ),
    }


@app.get("/credits/me")
async def credits_me(request: Request) -> dict[str, Any]:
    auth = await get_auth_context(request)
    return {"credits": await get_credit_summary(auth.client, auth.user["id"])}


@app.post("/credits/claim-trial")
async def claim_trial(request: Request) -> dict[str, Any]:
    auth = await get_auth_context(request)
    wallet, created = await ensure_credit_wallet(auth.client, auth.user["id"])
    return {
        "credits": await get_credit_summary(auth.client, auth.user["id"]),
        "message": "Free 100-credit trial activated." if created else "Free trial already active.",
        "wallet": wallet,
    }


@app.post("/billing/create-order")
async def create_billing_order(request: Request) -> dict[str, Any]:
    auth = await get_auth_context(request)
    body = await request.json()
    plan = resolve_plan(body.get("planId"))

    if plan["amount"] == 0:
        _, created = await ensure_credit_wallet(auth.client, auth.user["id"])
        return {
            "credits": await get_credit_summary(auth.client, auth.user["id"]),
            "message": "Free trial activated." if created else "Free trial is already active.",
            "plan": serialize_plan(plan),
        }

    key_id = os.environ.get("RAZORPAY_KEY_ID")
    key_secret = os.environ.get("RAZORPAY_KEY_SECRET")
    public_key = os.environ.get("NEXT_PUBLIC_RAZORPAY_KEY_ID") or key_id

    if not key_id or not key_secret or not public_key:
        raise api_error(503, "Razorpay keys are not configured yet.")

    receipt = f"ai-hiring-{uuid4().hex[:12]}"
    async with httpx.AsyncClient(auth=(key_id, key_secret), timeout=45.0) as client:
        response = await client.post(
            "https://api.razorpay.com/v1/orders",
            json={
                "amount": plan["amount"],
                "currency": plan["currency"],
                "notes": {
                    "credits": str(plan["credits"]),
                    "plan_id": plan["id"],
                    "user_id": auth.user["id"],
                },
                "receipt": receipt,
            },
        )
        response.raise_for_status()
        order = response.json()

    await auth.client.insert(
        "billing_orders",
        {
            "amount": plan["amount"],
            "created_at": now_iso(),
            "credits": plan["credits"],
            "currency": plan["currency"],
            "plan_id": plan["id"],
            "provider": "razorpay",
            "razorpay_order_id": order["id"],
            "receipt": receipt,
            "status": "created",
            "updated_at": now_iso(),
            "user_id": auth.user["id"],
        },
    )

    return {
        "keyId": public_key,
        "order": order,
        "plan": serialize_plan(plan),
    }


@app.post("/billing/verify")
async def verify_billing_order(request: Request) -> dict[str, Any]:
    auth = await get_auth_context(request)
    body = await request.json()

    order_id = clean_text(body.get("razorpay_order_id"), 120)
    payment_id = clean_text(body.get("razorpay_payment_id"), 120)
    signature = clean_text(body.get("razorpay_signature"), 256)

    if not order_id or not payment_id or not signature:
        raise api_error(400, "Incomplete Razorpay payment response.")

    order = await auth.client.maybe_single(
        "billing_orders",
        filters={"razorpay_order_id": order_id, "user_id": auth.user["id"]},
    )
    if not order:
        raise api_error(404, "Order not found for this user.")

    if order.get("status") == "paid":
        return {
            "credits": await get_credit_summary(auth.client, auth.user["id"]),
            "message": "Payment already verified.",
            "order": order,
        }

    secret = os.environ.get("RAZORPAY_KEY_SECRET")
    if not secret:
        raise api_error(503, "Razorpay secret is not configured.")

    generated = hmac.new(
        secret.encode("utf-8"),
        f"{order_id}|{payment_id}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(generated, signature):
        raise api_error(400, "Payment signature verification failed.")

    await auth.client.update(
        "billing_orders",
        {
            "razorpay_payment_id": payment_id,
            "status": "paid",
            "updated_at": now_iso(),
            "verified_at": now_iso(),
        },
        filters={"razorpay_order_id": order_id, "user_id": auth.user["id"]},
    )

    await add_credits(
        auth.client,
        auth.user["id"],
        int(order.get("credits") or 0),
        f"Razorpay top-up for {order.get('plan_id', 'credit pack')}",
        kind="purchase",
        metadata={"paymentId": payment_id, "provider": "razorpay"},
    )

    return {
        "credits": await get_credit_summary(auth.client, auth.user["id"]),
        "message": f"Payment verified and {order.get('credits', 0)} credits added.",
    }


@app.get("/candidate/profile")
async def candidate_profile(request: Request) -> dict[str, Any]:
    auth = await get_auth_context(request, "candidate")
    profile = await ensure_candidate_profile(auth.client, auth.user, auth.role)
    resume_bundle = await get_latest_resume_bundle(auth.client, auth.user["id"])
    recommended_jobs = await get_candidate_matches(auth.client, auth.user["id"])

    return {
        "credits": await get_credit_summary(auth.client, auth.user["id"]),
        "latestResume": resume_bundle["latestResume"],
        "parsingResult": resume_bundle["parsingResult"],
        "profile": {
            "bio": profile.get("bio") or "",
            "email": auth.user.get("email") or "",
            "headline": profile.get("headline") or "",
            "location": profile.get("location") or "",
            "name": display_name(auth.user),
            "profileCompletion": profile.get("profile_completion") or 0,
            "skills": profile.get("skills") or [],
            "yearsExperience": profile.get("years_experience") or 0,
        },
        "recommendedJobs": recommended_jobs,
        "resumeHistory": resume_bundle["history"],
    }


@app.patch("/candidate/profile")
async def update_candidate_profile(request: Request) -> dict[str, Any]:
    auth = await get_auth_context(request, "candidate")
    payload = parse_candidate_profile_payload(await request.json())
    current_profile = await ensure_candidate_profile(auth.client, auth.user, auth.role)

    await auth.client.update_auth_user(
        {
            "data": {
                **(auth.user.get("user_metadata") or {}),
                "full_name": payload["name"],
            }
        }
    )

    await auth.client.upsert(
        "users",
        {
            "email": auth.user.get("email"),
            "full_name": payload["name"],
            "id": auth.user["id"],
            "role": auth.role,
            "updated_at": now_iso(),
        },
        on_conflict="id",
    )

    merged_skills = unique_values([*(current_profile.get("skills") or []), *payload["skills"]])[:12]
    profile_completion = compute_profile_completion(
        {
            "bio": payload["bio"],
            "headline": payload["headline"],
            "latestResumeId": current_profile.get("latest_resume_id"),
            "location": payload["location"],
            "name": payload["name"],
            "skills": merged_skills,
            "yearsExperience": payload["yearsExperience"],
        }
    )

    await auth.client.update(
        "candidate_profiles",
        {
            "bio": payload["bio"],
            "headline": payload["headline"],
            "location": payload["location"],
            "profile_completion": profile_completion,
            "skills": merged_skills,
            "updated_at": now_iso(),
            "years_experience": payload["yearsExperience"],
        },
        filters={"user_id": auth.user["id"]},
    )

    await recompute_matches_for_candidate(auth.client, auth.user["id"])
    resume_bundle = await get_latest_resume_bundle(auth.client, auth.user["id"])
    recommended_jobs = await get_candidate_matches(auth.client, auth.user["id"])

    return {
        "credits": await get_credit_summary(auth.client, auth.user["id"]),
        "latestResume": resume_bundle["latestResume"],
        "message": "Profile updated successfully.",
        "parsingResult": resume_bundle["parsingResult"],
        "profile": {
            "bio": payload["bio"],
            "email": auth.user.get("email") or "",
            "headline": payload["headline"],
            "location": payload["location"],
            "name": payload["name"],
            "profileCompletion": profile_completion,
            "skills": merged_skills,
            "yearsExperience": payload["yearsExperience"],
        },
        "recommendedJobs": recommended_jobs,
        "resumeHistory": resume_bundle["history"],
    }


@app.get("/candidate/resume")
async def candidate_resume(request: Request) -> dict[str, Any]:
    auth = await get_auth_context(request, "candidate")
    await ensure_candidate_profile(auth.client, auth.user, auth.role)
    resume_bundle = await get_latest_resume_bundle(auth.client, auth.user["id"])
    recommended_jobs = await get_candidate_matches(auth.client, auth.user["id"])

    return {
        "credits": await get_credit_summary(auth.client, auth.user["id"]),
        "latestResume": resume_bundle["latestResume"],
        "model": {"modelVersion": MODEL_VERSION, **ml_service.export_metadata()},
        "parsingResult": resume_bundle["parsingResult"],
        "recommendedJobs": recommended_jobs,
        "resumeHistory": resume_bundle["history"],
    }


@app.post("/candidate/resume")
async def upload_candidate_resume(
    request: Request,
    file: UploadFile = File(...),
) -> dict[str, Any]:
    auth = await get_auth_context(request, "candidate")

    extension = Path(file.filename or "resume.txt").suffix.lower()
    if extension not in SUPPORTED_RESUME_EXTENSIONS:
        raise api_error(400, "Only PDF, DOCX, and TXT files are supported right now.")

    file_bytes = await file.read()
    if not file_bytes:
        raise api_error(400, "Please attach a resume file.")
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise api_error(400, "Please upload a file smaller than 8 MB.")

    await spend_credits(
        auth.client,
        auth.user["id"],
        ACTION_COSTS["resume_scan"],
        "ML resume analysis",
        metadata={"action": "resume_scan"},
    )

    existing_profile = await ensure_candidate_profile(auth.client, auth.user, auth.role)

    timestamp = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    file_name = file.filename or "resume.txt"
    file_path = f"{auth.user['id']}/{timestamp}-{sanitize_file_name(file_name)}"

    await auth.client.upload_file(
        bucket="resumes",
        path=file_path,
        content=file_bytes,
        content_type=file.content_type or "application/octet-stream",
        upsert=False,
    )

    resume_record = (
        await auth.client.insert(
            "resumes",
            {
                "file_name": file_name,
                "file_path": file_path,
                "file_size": len(file_bytes),
                "file_type": file.content_type or "application/octet-stream",
                "parsing_status": "processing",
                "updated_at": now_iso(),
                "user_id": auth.user["id"],
            },
        )
    )[0]

    try:
        extracted_text = ml_service.extract_text(
            file_bytes,
            file_name,
            file.content_type or "application/octet-stream",
        )
        insights = ml_service.predict_resume(extracted_text, file_name)
        merged_skills = unique_values([*(existing_profile.get("skills") or []), *insights.skills])[:12]
        headline = (existing_profile.get("headline") or "").strip() or insights.headline
        location = (existing_profile.get("location") or "").strip() or insights.location
        bio = (existing_profile.get("bio") or "").strip() or insights.summary
        years_experience = max(existing_profile.get("years_experience") or 0, insights.years_experience)

        await auth.client.upsert(
            "resume_parsing_results",
            {
                "category_confidence": insights.category_confidence,
                "certifications": insights.certifications,
                "created_at": now_iso(),
                "education": insights.education,
                "experience": insights.experience,
                "parsing_confidence": insights.confidence,
                "predicted_category": insights.category,
                "projects": insights.projects,
                "raw_text": insights.extracted_text,
                "resume_id": resume_record["id"],
                "skills": insights.skills,
                "suggestions": insights.suggestions,
                "summary": insights.summary,
            },
            on_conflict="resume_id",
        )

        await auth.client.update(
            "resumes",
            {
                "parsing_status": "completed",
                "updated_at": now_iso(),
            },
            filters={"id": resume_record["id"]},
        )

        await auth.client.update(
            "candidate_profiles",
            {
                "bio": bio,
                "headline": headline,
                "latest_resume_id": resume_record["id"],
                "location": location,
                "profile_completion": compute_profile_completion(
                    {
                        "bio": bio,
                        "headline": headline,
                        "latestResumeId": resume_record["id"],
                        "location": location,
                        "name": display_name(auth.user),
                        "skills": merged_skills,
                        "yearsExperience": years_experience,
                    }
                ),
                "skills": merged_skills,
                "updated_at": now_iso(),
                "years_experience": years_experience,
            },
            filters={"user_id": auth.user["id"]},
        )

        await recompute_matches_for_candidate(auth.client, auth.user["id"])
    except Exception:
        await auth.client.update(
            "resumes",
            {
                "parsing_status": "failed",
                "updated_at": now_iso(),
            },
            filters={"id": resume_record["id"]},
        )
        raise

    resume_bundle = await get_latest_resume_bundle(auth.client, auth.user["id"])
    recommended_jobs = await get_candidate_matches(auth.client, auth.user["id"])

    return {
        "credits": await get_credit_summary(auth.client, auth.user["id"]),
        "latestResume": resume_bundle["latestResume"],
        "message": f"Resume uploaded and scanned successfully. {ACTION_COSTS['resume_scan']} credits used.",
        "model": {"modelVersion": MODEL_VERSION, **ml_service.export_metadata()},
        "parsingResult": resume_bundle["parsingResult"],
        "recommendedJobs": recommended_jobs,
        "resumeHistory": resume_bundle["history"],
    }


@app.get("/candidate/applications")
async def candidate_applications(request: Request) -> dict[str, Any]:
    auth = await get_auth_context(request, "candidate")
    applications = await get_candidate_applications(auth.client, auth.user["id"])
    return {"applications": applications}


@app.post("/candidate/applications")
async def create_candidate_application(request: Request) -> dict[str, Any]:
    auth = await get_auth_context(request, "candidate")
    body = await request.json()
    job_id = clean_text(body.get("jobId"), 120)

    if not job_id:
        raise api_error(400, "Missing job id.")

    job = await auth.client.maybe_single(
        "jobs",
        filters={"id": job_id, "status": "active"},
    )
    if not job:
        raise api_error(404, "Job not found.")

    resume_bundle = await get_latest_resume_bundle(auth.client, auth.user["id"])
    latest_resume = resume_bundle["latestResume"]
    if not latest_resume:
        raise api_error(400, "Upload a resume before applying to jobs.")

    existing = await auth.client.maybe_single(
        "applications",
        filters={"candidate_id": auth.user["id"], "job_id": job_id},
    )
    if existing:
        return {
            "application": existing,
            "applications": await get_candidate_applications(auth.client, auth.user["id"]),
            "message": "You already applied to this job.",
        }

    application = (
        await auth.client.insert(
            "applications",
            {
                "candidate_id": auth.user["id"],
                "job_id": job_id,
                "resume_id": latest_resume["id"],
                "status": "submitted",
                "updated_at": now_iso(),
            },
        )
    )[0]

    return {
        "application": application,
        "applications": await get_candidate_applications(auth.client, auth.user["id"]),
        "message": "Application submitted successfully.",
    }


@app.get("/recruiter/profile")
async def recruiter_profile(request: Request) -> dict[str, Any]:
    auth = await get_auth_context(request, "recruiter")
    profile = await ensure_recruiter_profile(auth.client, auth.user, auth.role)
    return {
        "credits": await get_credit_summary(auth.client, auth.user["id"]),
        "profile": profile,
        "user": {
            "email": auth.user.get("email") or "",
            "name": display_name(auth.user),
        },
    }


@app.get("/recruiter/jobs")
async def recruiter_jobs(request: Request) -> dict[str, Any]:
    auth = await get_auth_context(request, "recruiter")
    await ensure_recruiter_profile(auth.client, auth.user, auth.role)
    jobs = await get_recruiter_jobs(auth.client, auth.user["id"])
    return {
        "credits": await get_credit_summary(auth.client, auth.user["id"]),
        "jobs": jobs,
    }


@app.post("/recruiter/jobs")
async def create_recruiter_job(request: Request) -> dict[str, Any]:
    auth = await get_auth_context(request, "recruiter")
    payload = parse_job_payload(await request.json())

    await ensure_recruiter_profile(auth.client, auth.user, auth.role)

    spend_amount = ACTION_COSTS["job_publish"] if payload["status"] == "active" else 0
    if spend_amount:
        await spend_credits(
            auth.client,
            auth.user["id"],
            spend_amount,
            "Publish recruiter job",
            metadata={"action": "job_publish"},
        )

    skills = build_job_skills(payload)
    job = (
        await auth.client.insert(
            "jobs",
            {
                "category": payload["category"],
                "description": payload["description"],
                "employment_type": payload["employmentType"],
                "location": payload["location"],
                "recruiter_id": auth.user["id"],
                "salary_max": payload["salaryMax"],
                "salary_min": payload["salaryMin"],
                "skills": skills,
                "status": payload["status"],
                "title": payload["title"],
                "updated_at": now_iso(),
            },
        )
    )[0]

    automation_updates = []
    message = "Job saved as draft."
    if payload["status"] == "active":
        automation_updates = await recompute_matches_for_job(auth.client, job["id"])
        message = f"Job published and AI matching refreshed. {spend_amount} credits used."

    return {
        "automationUpdatedCount": len(automation_updates),
        "credits": await get_credit_summary(auth.client, auth.user["id"]),
        "job": job,
        "jobs": await get_recruiter_jobs(auth.client, auth.user["id"]),
        "message": message,
    }


@app.get("/recruiter/dashboard")
async def recruiter_dashboard(request: Request) -> dict[str, Any]:
    auth = await get_auth_context(request, "recruiter")
    await ensure_recruiter_profile(auth.client, auth.user, auth.role)
    dashboard = await get_recruiter_dashboard_data(auth.client, auth.user["id"])

    return {
        "credits": await get_credit_summary(auth.client, auth.user["id"]),
        "jobs": dashboard["jobs"],
        "shortlist": dashboard["shortlist"],
        "stats": dashboard["stats"],
        "user": {
            "email": auth.user.get("email") or "",
            "name": display_name(auth.user),
        },
    }


@app.post("/automation/recompute")
async def automation_recompute(request: Request) -> dict[str, Any]:
    auth = await get_auth_context(request)
    content_type = request.headers.get("content-type", "")
    body = await request.json() if content_type.startswith("application/json") else {}
    scope = clean_text(body.get("scope"), 40) or "all"
    job_id = clean_text(body.get("jobId"), 120)

    if auth.role == "candidate":
        await spend_credits(
            auth.client,
            auth.user["id"],
            ACTION_COSTS["candidate_refresh"],
            "Refresh candidate AI matches",
            metadata={"action": "candidate_refresh"},
        )
        updated = await recompute_matches_for_candidate(auth.client, auth.user["id"])
        return {
            "credits": await get_credit_summary(auth.client, auth.user["id"]),
            "message": "Candidate matches refreshed.",
            "updatedCount": len(updated),
        }

    if scope == "job" and job_id:
        job = await auth.client.maybe_single(
            "jobs",
            filters={"id": job_id, "recruiter_id": auth.user["id"]},
            columns="id,title",
        )
        if not job:
            raise api_error(404, "Job not found for this recruiter.")

        await spend_credits(
            auth.client,
            auth.user["id"],
            ACTION_COSTS["job_refresh"],
            "Refresh one job shortlist",
            metadata={"action": "job_refresh", "jobId": job_id},
        )
        updated = await recompute_matches_for_job(auth.client, job_id)
        return {
            "credits": await get_credit_summary(auth.client, auth.user["id"]),
            "message": "Job matches refreshed.",
            "updatedCount": len(updated),
        }

    await spend_credits(
        auth.client,
        auth.user["id"],
        ACTION_COSTS["recruiter_refresh"],
        "Refresh recruiter AI dashboard",
        metadata={"action": "recruiter_refresh"},
    )
    updated = await recompute_matches_for_recruiter(auth.client, auth.user["id"])
    return {
        "credits": await get_credit_summary(auth.client, auth.user["id"]),
        "message": "Recruiter matches refreshed.",
        "updatedCount": len(updated),
    }


async def get_auth_context(
    request: Request,
    expected_role: str | None = None,
    required: bool = True,
) -> AuthContext | None:
    access_token = extract_access_token(request)
    if not access_token:
        if not required:
            return None
        raise api_error(401, "Missing access token.")

    client = SupabaseRestClient(access_token)
    user = await client.auth_user()
    role = normalize_role((user.get("user_metadata") or {}).get("role"))

    if not role:
        public_user = await client.maybe_single("users", columns="role", filters={"id": user["id"]})
        role = normalize_role(public_user.get("role") if public_user else None)

    if not role:
        raise api_error(403, "Your account role is missing. Please sign in again.")
    if expected_role and role != expected_role:
        raise api_error(403, "You do not have access to this resource.")

    await upsert_public_user(client, user, role)
    return AuthContext(access_token=access_token, client=client, role=role, user=user)


async def upsert_public_user(
    client: SupabaseRestClient,
    user: dict[str, Any],
    role: str,
) -> None:
    email = user.get("email")
    if not email:
        return

    await client.upsert(
        "users",
        {
            "avatar_url": (user.get("user_metadata") or {}).get("avatar_url"),
            "email": email,
            "full_name": display_name(user),
            "id": user["id"],
            "role": role,
            "updated_at": now_iso(),
        },
        on_conflict="id",
    )


async def ensure_candidate_profile(
    client: SupabaseRestClient,
    user: dict[str, Any],
    role: str,
) -> dict[str, Any]:
    await upsert_public_user(client, user, role)
    existing = await client.maybe_single("candidate_profiles", filters={"user_id": user["id"]})
    if existing:
        return existing

    inserted = (
        await client.insert(
            "candidate_profiles",
            {
                "bio": "",
                "headline": "",
                "location": "",
                "profile_completion": compute_profile_completion({"name": display_name(user)}),
                "skills": [],
                "user_id": user["id"],
                "years_experience": 0,
            },
        )
    )[0]
    return inserted


async def ensure_recruiter_profile(
    client: SupabaseRestClient,
    user: dict[str, Any],
    role: str,
) -> dict[str, Any]:
    await upsert_public_user(client, user, role)
    existing = await client.maybe_single("recruiter_profiles", filters={"user_id": user["id"]})
    if existing:
        return existing

    inserted = (
        await client.insert(
            "recruiter_profiles",
            {
                "company_name": "",
                "company_size": "",
                "industry": "",
                "user_id": user["id"],
                "website": "",
            },
        )
    )[0]
    return inserted


async def ensure_credit_wallet(
    client: SupabaseRestClient,
    user_id: str,
) -> tuple[dict[str, Any], bool]:
    existing = await client.maybe_single("credit_wallets", filters={"user_id": user_id})
    if existing:
        return existing, False

    payload = {
        "balance": FREE_TRIAL_CREDITS,
        "created_at": now_iso(),
        "total_purchased": 0,
        "total_spent": 0,
        "trial_claimed_at": now_iso(),
        "trial_credits": FREE_TRIAL_CREDITS,
        "updated_at": now_iso(),
        "user_id": user_id,
    }

    try:
        wallet = (await client.insert("credit_wallets", payload))[0]
        await client.insert(
            "credit_transactions",
            {
                "created_at": now_iso(),
                "delta": FREE_TRIAL_CREDITS,
                "description": "Free demo trial activated.",
                "kind": "trial",
                "metadata": {"source": "auto"},
                "user_id": user_id,
            },
        )
        return wallet, True
    except SupabaseError:
        wallet = await client.maybe_single("credit_wallets", filters={"user_id": user_id})
        if not wallet:
            raise
        return wallet, False


async def get_credit_summary(client: SupabaseRestClient, user_id: str) -> dict[str, Any]:
    wallet, _ = await ensure_credit_wallet(client, user_id)
    transactions = await client.select(
        "credit_transactions",
        columns="id,delta,kind,description,created_at,metadata",
        filters={"user_id": user_id},
        order=("created_at", False),
        limit=8,
    )
    return {
        "balance": wallet.get("balance") or 0,
        "recentTransactions": [
            {
                "createdAt": item.get("created_at"),
                "delta": item.get("delta") or 0,
                "description": item.get("description") or "",
                "id": item.get("id"),
                "kind": item.get("kind") or "activity",
                "metadata": item.get("metadata") or {},
            }
            for item in transactions
        ],
        "totalPurchased": wallet.get("total_purchased") or 0,
        "totalSpent": wallet.get("total_spent") or 0,
        "trialClaimedAt": wallet.get("trial_claimed_at"),
        "trialCredits": wallet.get("trial_credits") or FREE_TRIAL_CREDITS,
    }


async def spend_credits(
    client: SupabaseRestClient,
    user_id: str,
    amount: int,
    description: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if amount <= 0:
        wallet, _ = await ensure_credit_wallet(client, user_id)
        return wallet

    wallet, _ = await ensure_credit_wallet(client, user_id)
    balance = int(wallet.get("balance") or 0)
    if balance < amount:
        raise api_error(
            402,
            f"You need {amount} credits for this action. Please add credits on the pricing page.",
        )

    updated = (
        await client.update(
            "credit_wallets",
            {
                "balance": balance - amount,
                "total_spent": int(wallet.get("total_spent") or 0) + amount,
                "updated_at": now_iso(),
            },
            filters={"user_id": user_id},
        )
    )[0]

    await client.insert(
        "credit_transactions",
        {
            "created_at": now_iso(),
            "delta": -amount,
            "description": description,
            "kind": "spend",
            "metadata": metadata or {},
            "user_id": user_id,
        },
    )
    return updated


async def add_credits(
    client: SupabaseRestClient,
    user_id: str,
    amount: int,
    description: str,
    *,
    kind: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    wallet, _ = await ensure_credit_wallet(client, user_id)
    updated = (
        await client.update(
            "credit_wallets",
            {
                "balance": int(wallet.get("balance") or 0) + amount,
                "total_purchased": int(wallet.get("total_purchased") or 0) + amount,
                "updated_at": now_iso(),
            },
            filters={"user_id": user_id},
        )
    )[0]

    await client.insert(
        "credit_transactions",
        {
            "created_at": now_iso(),
            "delta": amount,
            "description": description,
            "kind": kind,
            "metadata": metadata or {},
            "user_id": user_id,
        },
    )
    return updated


async def get_latest_resume_bundle(
    client: SupabaseRestClient,
    user_id: str,
) -> dict[str, Any]:
    resumes = await client.select(
        "resumes",
        filters={"user_id": user_id},
        order=("uploaded_at", False),
        limit=5,
    )
    latest_resume = resumes[0] if resumes else None
    parsing_result = None

    if latest_resume:
        parsing_result = await client.maybe_single(
            "resume_parsing_results",
            filters={"resume_id": latest_resume["id"]},
        )

    return {
        "history": resumes,
        "latestResume": latest_resume,
        "parsingResult": parsing_result,
    }


async def get_candidate_matches(
    client: SupabaseRestClient,
    user_id: str,
    limit: int = 4,
) -> list[dict[str, Any]]:
    matches = await client.select(
        "match_results",
        columns="candidate_id,job_id,matched_skills,reason_summary,score",
        filters={"candidate_id": user_id},
        order=("score", False),
        limit=limit,
    )
    if not matches:
        return []

    job_ids = [match["job_id"] for match in matches]
    jobs = await client.select(
        "jobs",
        columns="id,recruiter_id,title,location,employment_type,salary_min,salary_max",
        filters={"id": ("in", job_ids)},
    )
    recruiter_ids = unique_values([job.get("recruiter_id") for job in jobs])
    recruiter_profiles = (
        await client.select(
            "recruiter_profiles",
            columns="company_name,user_id",
            filters={"user_id": ("in", recruiter_ids)},
        )
        if recruiter_ids
        else []
    )

    jobs_by_id = {job["id"]: job for job in jobs}
    recruiters_by_id = {item["user_id"]: item for item in recruiter_profiles}

    results = []
    for match in matches:
        job = jobs_by_id.get(match["job_id"], {})
        recruiter = recruiters_by_id.get(job.get("recruiter_id"), {})
        results.append(
            {
                "company": recruiter.get("company_name") or "Hiring team",
                "employmentType": job.get("employment_type") or "Full-time",
                "jobId": match["job_id"],
                "location": job.get("location") or "Not specified",
                "matchedSkills": match.get("matched_skills") or [],
                "reasonSummary": match.get("reason_summary") or "",
                "salaryMax": job.get("salary_max"),
                "salaryMin": job.get("salary_min"),
                "score": match.get("score") or 0,
                "title": job.get("title") or "Open role",
            }
        )
    return results


async def get_candidate_applications(
    client: SupabaseRestClient,
    user_id: str,
) -> list[dict[str, Any]]:
    applications = await client.select(
        "applications",
        columns="applied_at,id,job_id,resume_id,status",
        filters={"candidate_id": user_id},
        order=("applied_at", False),
    )
    if not applications:
        return []

    jobs = await client.select(
        "jobs",
        columns="id,title,location,employment_type",
        filters={"id": ("in", [application["job_id"] for application in applications])},
    )
    jobs_by_id = {job["id"]: job for job in jobs}

    return [
        {
            **application,
            "employmentType": jobs_by_id.get(application["job_id"], {}).get("employment_type")
            or "Full-time",
            "location": jobs_by_id.get(application["job_id"], {}).get("location")
            or "Not specified",
            "title": jobs_by_id.get(application["job_id"], {}).get("title") or "Open role",
        }
        for application in applications
    ]


async def get_public_jobs(client: SupabaseRestClient) -> list[dict[str, Any]]:
    jobs = await client.select(
        "jobs",
        columns=(
            "category,created_at,description,employment_type,id,location,recruiter_id,"
            "salary_max,salary_min,status,title,skills"
        ),
        filters={"status": "active"},
        order=("created_at", False),
    )
    recruiter_ids = unique_values([job.get("recruiter_id") for job in jobs])
    recruiters = (
        await client.select(
            "recruiter_profiles",
            columns="company_name,user_id",
            filters={"user_id": ("in", recruiter_ids)},
        )
        if recruiter_ids
        else []
    )
    companies = {item["user_id"]: item.get("company_name") for item in recruiters}

    return [
        {
            **job,
            "company_name": companies.get(job.get("recruiter_id")) or "Hiring team",
        }
        for job in jobs
    ]


async def get_recruiter_jobs(
    client: SupabaseRestClient,
    recruiter_id: str,
) -> list[dict[str, Any]]:
    jobs = await client.select(
        "jobs",
        filters={"recruiter_id": recruiter_id},
        order=("created_at", False),
    )
    if not jobs:
        return []

    job_ids = [job["id"] for job in jobs]
    applications = await client.select(
        "applications",
        columns="job_id,status",
        filters={"job_id": ("in", job_ids)},
    )
    matches = await client.select(
        "match_results",
        columns="job_id,score",
        filters={"job_id": ("in", job_ids)},
    )

    application_counts: dict[str, int] = {}
    top_scores: dict[str, int] = {}
    for application in applications:
        job_id = application.get("job_id")
        application_counts[job_id] = application_counts.get(job_id, 0) + 1
    for match in matches:
        job_id = match.get("job_id")
        top_scores[job_id] = max(int(match.get("score") or 0), top_scores.get(job_id, 0))

    return [
        {
            **job,
            "applicantCount": application_counts.get(job["id"], 0),
            "topMatchScore": top_scores.get(job["id"], 0),
        }
        for job in jobs
    ]


async def get_recruiter_dashboard_data(
    client: SupabaseRestClient,
    recruiter_id: str,
) -> dict[str, Any]:
    jobs = await get_recruiter_jobs(client, recruiter_id)
    job_ids = [job["id"] for job in jobs]

    applications = (
        await client.select(
            "applications",
            columns="candidate_id,job_id,status,applied_at",
            filters={"job_id": ("in", job_ids)},
        )
        if job_ids
        else []
    )
    matches = (
        await client.select(
            "match_results",
            columns="candidate_id,job_id,matched_skills,reason_summary,score",
            filters={"job_id": ("in", job_ids)},
            order=("score", False),
            limit=12,
        )
        if job_ids
        else []
    )

    candidate_ids = unique_values([match.get("candidate_id") for match in matches])
    candidate_profiles = (
        await client.select(
            "candidate_profiles",
            columns="bio,headline,location,skills,user_id,years_experience",
            filters={"user_id": ("in", candidate_ids)},
        )
        if candidate_ids
        else []
    )
    users = (
        await client.select(
            "users",
            columns="email,full_name,id",
            filters={"id": ("in", candidate_ids)},
        )
        if candidate_ids
        else []
    )

    profiles_by_id = {profile["user_id"]: profile for profile in candidate_profiles}
    users_by_id = {item["id"]: item for item in users}
    jobs_by_id = {job["id"]: job for job in jobs}

    shortlist = []
    for match in matches[:5]:
        profile = profiles_by_id.get(match.get("candidate_id"), {})
        user = users_by_id.get(match.get("candidate_id"), {})
        job = jobs_by_id.get(match.get("job_id"), {})
        shortlist.append(
            {
                "candidateId": match.get("candidate_id"),
                "email": user.get("email") or "",
                "headline": profile.get("headline") or "Candidate profile",
                "location": profile.get("location") or "Location not set",
                "matchedSkills": match.get("matched_skills") or [],
                "name": user.get("full_name") or user.get("email") or "Candidate",
                "reasonSummary": match.get("reason_summary") or "",
                "score": match.get("score") or 0,
                "targetRole": job.get("title") or "Open role",
                "yearsExperience": profile.get("years_experience") or 0,
            }
        )

    return {
        "jobs": jobs,
        "shortlist": shortlist,
        "stats": {
            "interviews": len([item for item in applications if item.get("status") == "interview"]),
            "newCandidates": len({item.get("candidate_id") for item in applications}),
            "openRoles": len([job for job in jobs if job.get("status") != "closed"]),
        },
    }


async def recompute_matches_for_candidate(
    client: SupabaseRestClient,
    candidate_id: str,
) -> list[dict[str, Any]]:
    profile = await client.maybe_single("candidate_profiles", filters={"user_id": candidate_id})
    if not profile:
        return []

    resume_bundle = await get_latest_resume_bundle(client, candidate_id)
    parsing = resume_bundle.get("parsingResult") or {}
    jobs = await client.select("jobs", filters={"status": "active"})
    if not jobs:
        return []

    resume_text = (parsing.get("raw_text") or "").strip() or build_profile_text(profile)
    resume_skills = unique_values([*(parsing.get("skills") or []), *(profile.get("skills") or [])])[:12]
    predicted_category = parsing.get("predicted_category") or (resume_skills[0] if resume_skills else "General")
    ranked = ml_service.rank_jobs(resume_text, resume_skills, predicted_category, jobs)

    payload = [
        {
            "candidate_id": candidate_id,
            "created_at": now_iso(),
            "job_id": item["jobId"],
            "matched_skills": item["matchedSkills"],
            "model_version": MODEL_VERSION,
            "reason_summary": item["reasonSummary"],
            "score": item["score"],
            "updated_at": now_iso(),
        }
        for item in ranked
    ]
    if payload:
        await client.upsert("match_results", payload, on_conflict="job_id,candidate_id")
    return payload


async def recompute_matches_for_job(
    client: SupabaseRestClient,
    job_id: str,
) -> list[dict[str, Any]]:
    job = await client.maybe_single("jobs", filters={"id": job_id})
    if not job or job.get("status") != "active":
        return []

    candidates = await client.select("candidate_profiles")
    if not candidates:
        return []

    payload = []
    for candidate in candidates:
        resume_bundle = await get_latest_resume_bundle(client, candidate["user_id"])
        parsing = resume_bundle.get("parsingResult") or {}
        resume_text = (parsing.get("raw_text") or "").strip() or build_profile_text(candidate)
        resume_skills = unique_values([*(parsing.get("skills") or []), *(candidate.get("skills") or [])])[:12]
        predicted_category = parsing.get("predicted_category") or (resume_skills[0] if resume_skills else "General")
        ranked = ml_service.rank_jobs(resume_text, resume_skills, predicted_category, [job])
        if not ranked:
            continue
        item = ranked[0]
        payload.append(
            {
                "candidate_id": candidate["user_id"],
                "created_at": now_iso(),
                "job_id": job_id,
                "matched_skills": item["matchedSkills"],
                "model_version": MODEL_VERSION,
                "reason_summary": item["reasonSummary"],
                "score": item["score"],
                "updated_at": now_iso(),
            }
        )

    if payload:
        await client.upsert("match_results", payload, on_conflict="job_id,candidate_id")
    return payload


async def recompute_matches_for_recruiter(
    client: SupabaseRestClient,
    recruiter_id: str,
) -> list[dict[str, Any]]:
    jobs = await client.select(
        "jobs",
        columns="id",
        filters={"recruiter_id": recruiter_id, "status": "active"},
    )
    updated: list[dict[str, Any]] = []
    for job in jobs:
        updated.extend(await recompute_matches_for_job(client, job["id"]))
    return updated


def extract_access_token(request: Request) -> str | None:
    authorization = request.headers.get("authorization") or ""
    if not authorization.startswith("Bearer "):
        return None
    return authorization.removeprefix("Bearer ").strip() or None


def api_error(status: int, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"error": message})


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_role(role: Any) -> str | None:
    if role in {"candidate", "recruiter", "admin"}:
        return str(role)
    return None


def display_name(user: dict[str, Any]) -> str:
    metadata = user.get("user_metadata") or {}
    return (
        metadata.get("full_name")
        or metadata.get("name")
        or (user.get("email") or "user").split("@")[0]
    )


def clean_text(value: Any, max_length: int) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()[:max_length]


def parse_number(value: Any, fallback: int = 0) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str) and value.strip():
        try:
            return int(float(value))
        except ValueError:
            return fallback
    return fallback


def parse_optional_number(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    parsed = parse_number(value, fallback=-1)
    return parsed if parsed >= 0 else None


def normalize_skills(value: Any) -> list[str]:
    if isinstance(value, list):
        items = [clean_text(item, 40) for item in value]
    elif isinstance(value, str):
        items = [clean_text(item, 40) for item in value.split(",")]
    else:
        items = []
    return unique_values([item.replace("  ", " ") for item in items if item])[:12]


def parse_candidate_profile_payload(payload: dict[str, Any]) -> dict[str, Any]:
    name = clean_text(payload.get("name"), 120)
    if not name:
        raise api_error(400, "Name is required.")

    return {
        "bio": clean_text(payload.get("bio"), 1200),
        "headline": clean_text(payload.get("headline"), 120),
        "location": clean_text(payload.get("location"), 120),
        "name": name,
        "skills": normalize_skills(payload.get("skills")),
        "yearsExperience": max(0, min(40, parse_number(payload.get("yearsExperience"), 0))),
    }


def parse_job_payload(payload: dict[str, Any]) -> dict[str, Any]:
    title = clean_text(payload.get("title"), 140)
    description = clean_text(payload.get("description"), 5000)
    if not title:
        raise api_error(400, "Job title is required.")
    if not description:
        raise api_error(400, "Job description is required.")

    return {
        "category": clean_text(payload.get("category"), 80),
        "description": description,
        "employmentType": clean_text(payload.get("employmentType"), 80) or "Full-time",
        "location": clean_text(payload.get("location"), 120),
        "salaryMax": parse_optional_number(payload.get("salaryMax")),
        "salaryMin": parse_optional_number(payload.get("salaryMin")),
        "status": clean_text(payload.get("status"), 40) or "active",
        "title": title,
    }


def compute_profile_completion(input_data: dict[str, Any]) -> int:
    checks = [
        bool(input_data.get("name")),
        bool(input_data.get("headline")),
        bool(input_data.get("location")),
        bool(input_data.get("bio")),
        bool(input_data.get("latestResumeId")),
        bool(input_data.get("skills")),
        bool((input_data.get("yearsExperience") or 0) > 0),
    ]
    return round((len([item for item in checks if item]) / len(checks)) * 100)


def unique_values(items: list[Any]) -> list[Any]:
    output: list[Any] = []
    for item in items:
        if item and item not in output:
            output.append(item)
    return output


def sanitize_file_name(file_name: str) -> str:
    safe = file_name.lower().replace(" ", "-")
    return "".join(char for char in safe if char.isalnum() or char in {"-", "."})


def build_job_skills(payload: dict[str, Any]) -> list[str]:
    text = " ".join(
        [payload.get("title") or "", payload.get("category") or "", payload.get("description") or ""]
    )
    return extract_skills(text, [])


def build_profile_text(profile: dict[str, Any]) -> str:
    return " ".join(
        [
            profile.get("headline") or "",
            profile.get("bio") or "",
            " ".join(profile.get("skills") or []),
            profile.get("location") or "",
        ]
    ).strip()


def resolve_plan(plan_id: Any) -> dict[str, Any]:
    for plan in PRICING_PLANS:
        if plan["id"] == plan_id:
            return plan
    raise api_error(404, "Unknown pricing plan.")


def serialize_plan(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        **plan,
        "amountDisplay": f"{plan['currency']} {plan['amount'] / 100:.0f}" if plan["amount"] else "Free",
    }
