from __future__ import annotations

import json
import hashlib
import hmac
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus
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

AI_HIRING_PROFILE_KEY = "ai_hiring_profile"
AI_HIRING_RESUME_KEY = "ai_hiring_resume"
COMPATIBILITY_STORE_PATH = Path(__file__).resolve().parent / "data" / "compatibility_store.json"
COMPATIBILITY_STORE_LOCK = threading.Lock()
COMPATIBILITY_MODE_MESSAGE = (
    "The AI Hiring Supabase schema is not installed in this project yet. "
    "Running in compatibility mode with limited storage."
)

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

PORTFOLIO_PROFILE = {
    "availability": "Open for new projects",
    "email": "divy9anshu@gmail.com",
    "headline": "Full Stack Developer & AI Builder",
    "name": "Divyanshu",
    "response_time": "Most inquiries get a response within 2 hours.",
    "whatsapp": "+91 9334805955",
    "whatsapp_url": "https://wa.me/919334805955",
    "work_regions": ["UK", "USA", "Canada", "Worldwide"],
    "years_experience": "3+",
}


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


@app.get("/about/profile")
async def about_profile() -> dict[str, Any]:
    return {"profile": PORTFOLIO_PROFILE}


@app.post("/about/contact")
async def create_about_contact_request(request: Request) -> dict[str, Any]:
    authenticated = await get_authenticated_user(request)
    user = authenticated[1] if authenticated else None
    client = authenticated[0] if authenticated else SupabaseRestClient()
    payload = parse_contact_request_payload(await request.json(), user)
    timestamp = now_iso()
    record = {
        "budget": payload["budget"],
        "created_at": timestamp,
        "email": payload["email"],
        "message": payload["message"],
        "name": payload["name"],
        "source_page": payload["sourcePage"],
        "status": "new",
        "updated_at": timestamp,
        "user_id": user.get("id") if user else None,
    }

    try:
        contact_request = (await client.insert("contact_requests", record))[0]
    except SupabaseError as exc:
        if not is_missing_table_error(exc):
            raise
        contact_request = create_compatibility_contact_request(record)

    return {
        "contact": PORTFOLIO_PROFILE,
        "contactRequest": contact_request,
        "message": (
            "Thanks, your message was saved. You can also reach out on WhatsApp for a faster reply."
        ),
    }


@app.post("/consultations/request")
async def request_consultation(request: Request) -> dict[str, Any]:
    authenticated = await get_authenticated_user(request)
    user = authenticated[1] if authenticated else None
    client = authenticated[0] if authenticated else SupabaseRestClient()
    payload = parse_consultation_request_payload(await request.json(), user)
    timestamp = now_iso()
    record = {
        "company_name": payload["companyName"],
        "created_at": timestamp,
        "email": payload["email"],
        "message": payload["message"],
        "name": payload["name"],
        "source_page": payload["sourcePage"],
        "status": "pending",
        "team_size": payload["teamSize"],
        "updated_at": timestamp,
        "user_id": user.get("id") if user else None,
    }

    try:
        consultation = (await client.insert("consultation_requests", record))[0]
    except SupabaseError as exc:
        if not is_missing_table_error(exc):
            raise
        consultation = create_compatibility_consultation_request(record)

    return {
        "consultation": consultation,
        "message": "Consultation request received. Our team will contact you soon.",
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

    timestamp = now_iso()
    try:
        await auth.client.insert(
            "billing_orders",
            {
                "amount": plan["amount"],
                "created_at": timestamp,
                "credits": plan["credits"],
                "currency": plan["currency"],
                "plan_id": plan["id"],
                "provider": "razorpay",
                "razorpay_order_id": order["id"],
                "receipt": receipt,
                "status": "created",
                "updated_at": timestamp,
                "user_id": auth.user["id"],
            },
        )
    except SupabaseError as exc:
        if not is_missing_table_error(exc):
            raise
        create_compatibility_billing_order(
            {
                "amount": plan["amount"],
                "created_at": timestamp,
                "credits": plan["credits"],
                "currency": plan["currency"],
                "plan_id": plan["id"],
                "provider": "razorpay",
                "razorpay_order_id": order["id"],
                "receipt": receipt,
                "status": "created",
                "updated_at": timestamp,
                "user_id": auth.user["id"],
            }
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

    try:
        order = await auth.client.maybe_single(
            "billing_orders",
            filters={"razorpay_order_id": order_id, "user_id": auth.user["id"]},
        )
    except SupabaseError as exc:
        if not is_missing_table_error(exc):
            raise
        order = get_compatibility_billing_order(auth.user["id"], order_id)
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

    verified_at = now_iso()
    try:
        await auth.client.update(
            "billing_orders",
            {
                "razorpay_payment_id": payment_id,
                "status": "paid",
                "updated_at": verified_at,
                "verified_at": verified_at,
            },
            filters={"razorpay_order_id": order_id, "user_id": auth.user["id"]},
        )
    except SupabaseError as exc:
        if not is_missing_table_error(exc):
            raise
        updated_order = update_compatibility_billing_order(
            auth.user["id"],
            order_id,
            {
                "razorpay_payment_id": payment_id,
                "status": "paid",
                "updated_at": verified_at,
                "verified_at": verified_at,
            },
        )
        if not updated_order:
            raise api_error(404, "Order not found for this user.")

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
    resume_bundle = await get_latest_resume_bundle(
        auth.client, auth.user["id"], auth.user
    )
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
    compatibility_mode = bool(current_profile.get("__compatibility_mode"))
    resume_bundle = await get_latest_resume_bundle(auth.client, auth.user["id"], auth.user)
    merged_skills = unique_values([*(current_profile.get("skills") or []), *payload["skills"]])[:12]
    profile_completion = compute_profile_completion(
        {
            "bio": payload["bio"],
            "headline": payload["headline"],
            "latestResumeId": (
                (resume_bundle.get("latestResume") or {}).get("id")
                or current_profile.get("latest_resume_id")
            ),
            "location": payload["location"],
            "name": payload["name"],
            "skills": merged_skills,
            "yearsExperience": payload["yearsExperience"],
        }
    )
    updated_metadata = {
        **(auth.user.get("user_metadata") or {}),
        "full_name": payload["name"],
    }

    if compatibility_mode:
        updated_metadata[AI_HIRING_PROFILE_KEY] = {
            "bio": payload["bio"],
            "headline": payload["headline"],
            "location": payload["location"],
            "profile_completion": profile_completion,
            "role": auth.role,
            "skills": merged_skills,
            "years_experience": payload["yearsExperience"],
        }

    await auth.client.update_auth_user({"data": updated_metadata})
    updated_user = {**auth.user, "user_metadata": updated_metadata}

    if not compatibility_mode:
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

    resume_bundle = await get_latest_resume_bundle(
        auth.client, auth.user["id"], updated_user
    )
    recommended_jobs = (
        []
        if compatibility_mode
        else await get_candidate_matches(auth.client, auth.user["id"])
    )

    return {
        "credits": await get_credit_summary(auth.client, auth.user["id"]),
        "latestResume": resume_bundle["latestResume"],
        "message": (
            "Profile updated in compatibility mode. Apply the AI Hiring Supabase schema to unlock saved resumes, jobs, and AI matching."
            if compatibility_mode
            else "Profile updated successfully."
        ),
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
    resume_bundle = await get_latest_resume_bundle(
        auth.client, auth.user["id"], auth.user
    )
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

    existing_profile = await ensure_candidate_profile(auth.client, auth.user, auth.role)
    compatibility_mode = bool(existing_profile.get("__compatibility_mode"))

    if compatibility_mode:
        extracted_text = ml_service.extract_text(
            file_bytes,
            file.filename or "resume.txt",
            file.content_type or "application/octet-stream",
        )
        insights = ml_service.predict_resume(extracted_text, file.filename or "resume.txt")
        timestamp_iso = now_iso()
        merged_skills = unique_values([*(existing_profile.get("skills") or []), *insights.skills])[:12]
        headline = (existing_profile.get("headline") or "").strip() or insights.headline
        location = (existing_profile.get("location") or "").strip() or insights.location
        bio = (existing_profile.get("bio") or "").strip() or insights.summary
        years_experience = max(existing_profile.get("years_experience") or 0, insights.years_experience)
        resume_id = f"compat-{auth.user['id']}-{int(datetime.now(tz=timezone.utc).timestamp())}"
        profile_completion = compute_profile_completion(
            {
                "bio": bio,
                "headline": headline,
                "latestResumeId": resume_id,
                "location": location,
                "name": display_name(auth.user),
                "skills": merged_skills,
                "yearsExperience": years_experience,
            }
        )

        updated_metadata = {
            **(auth.user.get("user_metadata") or {}),
            AI_HIRING_PROFILE_KEY: {
                "bio": bio,
                "headline": headline,
                "location": location,
                "profile_completion": profile_completion,
                "role": auth.role,
                "skills": merged_skills,
                "years_experience": years_experience,
            },
            AI_HIRING_RESUME_KEY: {
                "file_name": file.filename or "resume.txt",
                "id": resume_id,
                "parsing_status": "completed",
                "skills": insights.skills,
                "suggestions": insights.suggestions,
                "summary": insights.summary,
                "uploaded_at": timestamp_iso,
            },
        }

        await auth.client.update_auth_user({"data": updated_metadata})
        updated_user = {**auth.user, "user_metadata": updated_metadata}
        resume_bundle = await get_latest_resume_bundle(
            auth.client, auth.user["id"], updated_user
        )

        return {
            "credits": compatibility_credit_summary(auth.user["id"]),
            "latestResume": resume_bundle["latestResume"],
            "message": "Resume analyzed in compatibility mode. Apply the AI Hiring Supabase schema to save resume history and AI job matches.",
            "model": {"modelVersion": MODEL_VERSION, **ml_service.export_metadata()},
            "parsingResult": resume_bundle["parsingResult"],
            "recommendedJobs": [],
            "resumeHistory": resume_bundle["history"],
        }

    await spend_credits(
        auth.client,
        auth.user["id"],
        ACTION_COSTS["resume_scan"],
        "ML resume analysis",
        metadata={"action": "resume_scan"},
    )

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

    resume_bundle = await get_latest_resume_bundle(
        auth.client, auth.user["id"], auth.user
    )
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

    resume_bundle = await get_latest_resume_bundle(
        auth.client,
        auth.user["id"],
        auth.user,
    )
    latest_resume = resume_bundle["latestResume"]
    if not latest_resume:
        raise api_error(400, "Upload a resume before applying to jobs.")

    try:
        job = await auth.client.maybe_single(
            "jobs",
            filters={"id": job_id, "status": "active"},
        )
        if not job:
            raise api_error(404, "Job not found.")

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
    except SupabaseError as exc:
        if not is_missing_table_error(exc):
            raise

        application, created = create_compatibility_application(
            auth.user["id"],
            job_id,
            latest_resume.get("id"),
        )
        return {
            "application": application,
            "applications": get_compatibility_candidate_applications(auth.user["id"]),
            "message": (
                "Application submitted in compatibility mode."
                if created
                else "You already applied to this job."
            ),
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

    recruiter_profile = await ensure_recruiter_profile(auth.client, auth.user, auth.role)

    try:
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
    except SupabaseError as exc:
        if not is_missing_table_error(exc):
            raise
        job = create_compatibility_job(
            auth.user["id"],
            payload,
            recruiter_profile,
            display_name(auth.user),
        )
        message = "Job saved as draft in compatibility mode."
        if payload["status"] == "active":
            message = (
                f"Job published in compatibility mode. {ACTION_COSTS['job_publish']} credits used."
            )
        return {
            "automationUpdatedCount": 0,
            "credits": await get_credit_summary(auth.client, auth.user["id"]),
            "job": job,
            "jobs": list_compatibility_jobs(auth.user["id"]),
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
        "recentJobs": dashboard["recentJobs"],
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
        try:
            public_user = await client.maybe_single(
                "users",
                columns="role",
                filters={"id": user["id"]},
            )
        except SupabaseError as exc:
            if not is_missing_table_error(exc, "users"):
                raise
            public_user = None
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

    try:
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
    except SupabaseError as exc:
        if not is_missing_table_error(exc, "users"):
            raise


async def ensure_candidate_profile(
    client: SupabaseRestClient,
    user: dict[str, Any],
    role: str,
) -> dict[str, Any]:
    await upsert_public_user(client, user, role)
    try:
        existing = await client.maybe_single(
            "candidate_profiles", filters={"user_id": user["id"]}
        )
        if existing:
            return existing

        inserted = (
            await client.insert(
                "candidate_profiles",
                {
                    "bio": "",
                    "headline": "",
                    "location": "",
                    "profile_completion": compute_profile_completion(
                        {"name": display_name(user)}
                    ),
                    "skills": [],
                    "user_id": user["id"],
                    "years_experience": 0,
                },
            )
        )[0]
        return inserted
    except SupabaseError as exc:
        if not is_missing_table_error(exc):
            raise
        return build_compatibility_candidate_profile(user, role)


async def ensure_recruiter_profile(
    client: SupabaseRestClient,
    user: dict[str, Any],
    role: str,
) -> dict[str, Any]:
    await upsert_public_user(client, user, role)
    try:
        existing = await client.maybe_single(
            "recruiter_profiles", filters={"user_id": user["id"]}
        )
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
    except SupabaseError as exc:
        if not is_missing_table_error(exc):
            raise
        return build_compatibility_recruiter_profile(user, role)


async def ensure_credit_wallet(
    client: SupabaseRestClient,
    user_id: str,
) -> tuple[dict[str, Any], bool]:
    try:
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
    except SupabaseError as exc:
        if not is_missing_table_error(exc):
            raise
        return ensure_compatibility_credit_wallet(user_id)


async def get_credit_summary(client: SupabaseRestClient, user_id: str) -> dict[str, Any]:
    try:
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
    except SupabaseError as exc:
        if not is_missing_table_error(exc):
            raise
        return compatibility_credit_summary(user_id)


async def spend_credits(
    client: SupabaseRestClient,
    user_id: str,
    amount: int,
    description: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
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
    except SupabaseError as exc:
        if not is_missing_table_error(exc):
            raise
        return spend_compatibility_credits(user_id, amount, description, metadata)


async def add_credits(
    client: SupabaseRestClient,
    user_id: str,
    amount: int,
    description: str,
    *,
    kind: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
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
    except SupabaseError as exc:
        if not is_missing_table_error(exc):
            raise
        return add_compatibility_credits(user_id, amount, description, kind, metadata)


async def get_latest_resume_bundle(
    client: SupabaseRestClient,
    user_id: str,
    user: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
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
    except SupabaseError as exc:
        if not is_missing_table_error(exc):
            raise
        return build_compatibility_resume_bundle(user)


async def get_candidate_matches(
    client: SupabaseRestClient,
    user_id: str,
    limit: int = 4,
) -> list[dict[str, Any]]:
    try:
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
    except SupabaseError as exc:
        if not is_missing_table_error(exc):
            raise
        return []


async def get_candidate_applications(
    client: SupabaseRestClient,
    user_id: str,
) -> list[dict[str, Any]]:
    try:
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
    except SupabaseError as exc:
        if not is_missing_table_error(exc):
            raise
        return get_compatibility_candidate_applications(user_id)


async def get_public_jobs(client: SupabaseRestClient) -> list[dict[str, Any]]:
    try:
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
        recruiters = []
        if recruiter_ids:
            try:
                recruiters = await client.select(
                    "recruiter_profiles",
                    columns="company_name,user_id",
                    filters={"user_id": ("in", recruiter_ids)},
                )
            except SupabaseError as exc:
                if not is_missing_table_error(exc):
                    raise
        companies = {item["user_id"]: item.get("company_name") for item in recruiters}

        return [
            {
                **job,
                "company_name": companies.get(job.get("recruiter_id")) or "Hiring team",
                "google_search_url": build_google_job_search_url(
                    job.get("title") or "",
                    job.get("location") or "",
                ),
                "linkedin_search_url": build_linkedin_job_search_url(
                    job.get("title") or "",
                    job.get("location") or "",
                ),
            }
            for job in jobs
        ]
    except SupabaseError as exc:
        if not is_missing_table_error(exc):
            raise
        return list_compatibility_jobs(only_active=True)


async def get_recruiter_jobs(
    client: SupabaseRestClient,
    recruiter_id: str,
) -> list[dict[str, Any]]:
    try:
        jobs = await client.select(
            "jobs",
            filters={"recruiter_id": recruiter_id},
            order=("created_at", False),
        )
        if not jobs:
            return []

        job_ids = [job["id"] for job in jobs]
        applications = []
        matches = []

        try:
            applications = await client.select(
                "applications",
                columns="job_id,status",
                filters={"job_id": ("in", job_ids)},
            )
        except SupabaseError as exc:
            if not is_missing_table_error(exc):
                raise

        try:
            matches = await client.select(
                "match_results",
                columns="job_id,score",
                filters={"job_id": ("in", job_ids)},
            )
        except SupabaseError as exc:
            if not is_missing_table_error(exc):
                raise

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
                "google_search_url": build_google_job_search_url(
                    job.get("title") or "",
                    job.get("location") or "",
                ),
                "linkedin_search_url": build_linkedin_job_search_url(
                    job.get("title") or "",
                    job.get("location") or "",
                ),
                "topMatchScore": top_scores.get(job["id"], 0),
            }
            for job in jobs
        ]
    except SupabaseError as exc:
        if not is_missing_table_error(exc):
            raise
        return list_compatibility_jobs(recruiter_id)


async def get_recruiter_dashboard_data(
    client: SupabaseRestClient,
    recruiter_id: str,
) -> dict[str, Any]:
    try:
        jobs = await get_recruiter_jobs(client, recruiter_id)
        recent_jobs = await get_public_jobs(client)
        job_ids = [job["id"] for job in jobs]

        applications = []
        matches = []
        if job_ids:
            try:
                applications = await client.select(
                    "applications",
                    columns="candidate_id,job_id,status,applied_at",
                    filters={"job_id": ("in", job_ids)},
                )
            except SupabaseError as exc:
                if not is_missing_table_error(exc):
                    raise

            try:
                matches = await client.select(
                    "match_results",
                    columns="candidate_id,job_id,matched_skills,reason_summary,score",
                    filters={"job_id": ("in", job_ids)},
                    order=("score", False),
                    limit=12,
                )
            except SupabaseError as exc:
                if not is_missing_table_error(exc):
                    raise

        candidate_ids = unique_values([match.get("candidate_id") for match in matches])
        candidate_profiles = []
        users = []
        if candidate_ids:
            try:
                candidate_profiles = await client.select(
                    "candidate_profiles",
                    columns="bio,headline,location,skills,user_id,years_experience",
                    filters={"user_id": ("in", candidate_ids)},
                )
            except SupabaseError as exc:
                if not is_missing_table_error(exc):
                    raise

            try:
                users = await client.select(
                    "users",
                    columns="email,full_name,id",
                    filters={"id": ("in", candidate_ids)},
                )
            except SupabaseError as exc:
                if not is_missing_table_error(exc):
                    raise

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
            "recentJobs": recent_jobs,
            "shortlist": shortlist,
            "stats": {
                "interviews": len(
                    [item for item in applications if item.get("status") == "interview"]
                ),
                "newCandidates": len({item.get("candidate_id") for item in applications}),
                "openRoles": len([job for job in jobs if job.get("status") != "closed"]),
            },
        }
    except SupabaseError as exc:
        if not is_missing_table_error(exc):
            raise
        return get_compatibility_recruiter_dashboard(recruiter_id)


async def recompute_matches_for_candidate(
    client: SupabaseRestClient,
    candidate_id: str,
) -> list[dict[str, Any]]:
    try:
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
    except SupabaseError as exc:
        if not is_missing_table_error(exc):
            raise
        return []


async def recompute_matches_for_job(
    client: SupabaseRestClient,
    job_id: str,
) -> list[dict[str, Any]]:
    try:
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
    except SupabaseError as exc:
        if not is_missing_table_error(exc):
            raise
        return []


async def recompute_matches_for_recruiter(
    client: SupabaseRestClient,
    recruiter_id: str,
) -> list[dict[str, Any]]:
    try:
        jobs = await client.select(
            "jobs",
            columns="id",
            filters={"recruiter_id": recruiter_id, "status": "active"},
        )
        updated: list[dict[str, Any]] = []
        for job in jobs:
            updated.extend(await recompute_matches_for_job(client, job["id"]))
        return updated
    except SupabaseError as exc:
        if not is_missing_table_error(exc):
            raise
        return []


async def get_authenticated_user(
    request: Request,
) -> tuple[SupabaseRestClient, dict[str, Any]] | None:
    access_token = extract_access_token(request)
    if not access_token:
        return None

    client = SupabaseRestClient(access_token)
    try:
        user = await client.auth_user()
    except SupabaseError:
        return None
    return client, user


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


def is_missing_table_error(error: Exception, table: str | None = None) -> bool:
    message = str(error).lower()
    if "could not find the table" not in message:
        return False
    if table is None:
        return True
    return f"public.{table}".lower() in message


def _empty_compatibility_store() -> dict[str, Any]:
    return {
        "applications": {},
        "billing_orders": {},
        "contact_requests": {},
        "consultation_requests": {},
        "credit_transactions": {},
        "credit_wallets": {},
        "jobs": {},
    }


def _load_compatibility_store_unlocked() -> dict[str, Any]:
    store = _empty_compatibility_store()
    if not COMPATIBILITY_STORE_PATH.exists():
        return store

    try:
        payload = json.loads(COMPATIBILITY_STORE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return store

    if not isinstance(payload, dict):
        return store

    for key in store:
        value = payload.get(key)
        if isinstance(value, dict):
            store[key] = value

    return store


def _save_compatibility_store_unlocked(store: dict[str, Any]) -> None:
    COMPATIBILITY_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = COMPATIBILITY_STORE_PATH.with_suffix(".tmp")
    temp_path.write_text(json.dumps(store, indent=2), encoding="utf-8")
    temp_path.replace(COMPATIBILITY_STORE_PATH)


def build_linkedin_job_search_url(title: str, location: str | None = None) -> str:
    keywords = clean_text(title, 140) or "job"
    location_text = clean_text(location, 120)
    url = f"https://www.linkedin.com/jobs/search/?keywords={quote_plus(keywords)}"
    if location_text:
        url += f"&location={quote_plus(location_text)}"
    return url


def build_google_job_search_url(title: str, location: str | None = None) -> str:
    keywords = clean_text(title, 140) or "job"
    location_text = clean_text(location, 120)
    query = f"{keywords} recent jobs"
    if location_text:
        query += f" {location_text}"
    return f"https://www.google.com/search?q={quote_plus(query)}&ibp=htl%3Bjobs"


def build_compatibility_credit_wallet(
    user_id: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = payload or {}
    return {
        "balance": parse_number(payload.get("balance"), FREE_TRIAL_CREDITS),
        "created_at": payload.get("created_at"),
        "id": payload.get("id") or f"compat-wallet-{user_id}",
        "total_purchased": parse_number(payload.get("total_purchased"), 0),
        "total_spent": parse_number(payload.get("total_spent"), 0),
        "trial_claimed_at": payload.get("trial_claimed_at"),
        "trial_credits": parse_number(payload.get("trial_credits"), FREE_TRIAL_CREDITS),
        "updated_at": payload.get("updated_at"),
        "user_id": user_id,
    }


def _ensure_compatibility_wallet_unlocked(
    store: dict[str, Any],
    user_id: str,
) -> tuple[dict[str, Any], bool]:
    wallets = store["credit_wallets"]
    transactions = store["credit_transactions"]

    existing = wallets.get(user_id)
    if isinstance(existing, dict):
        wallet = build_compatibility_credit_wallet(user_id, existing)
        wallets[user_id] = wallet
        user_transactions = transactions.get(user_id)
        transactions[user_id] = user_transactions if isinstance(user_transactions, list) else []
        return wallet, False

    timestamp = now_iso()
    wallet = build_compatibility_credit_wallet(
        user_id,
        {
            "balance": FREE_TRIAL_CREDITS,
            "created_at": timestamp,
            "trial_claimed_at": timestamp,
            "trial_credits": FREE_TRIAL_CREDITS,
            "updated_at": timestamp,
        },
    )
    wallets[user_id] = wallet
    transactions[user_id] = [
        {
            "created_at": timestamp,
            "delta": FREE_TRIAL_CREDITS,
            "description": "Free demo trial activated.",
            "id": f"compat-credit-{uuid4().hex[:12]}",
            "kind": "trial",
            "metadata": {"source": "compatibility"},
        }
    ]
    return wallet, True


def ensure_compatibility_credit_wallet(user_id: str) -> tuple[dict[str, Any], bool]:
    with COMPATIBILITY_STORE_LOCK:
        store = _load_compatibility_store_unlocked()
        wallet, created = _ensure_compatibility_wallet_unlocked(store, user_id)
        if created:
            _save_compatibility_store_unlocked(store)
        return wallet, created


def compatibility_credit_summary(user_id: str) -> dict[str, Any]:
    with COMPATIBILITY_STORE_LOCK:
        store = _load_compatibility_store_unlocked()
        wallet, created = _ensure_compatibility_wallet_unlocked(store, user_id)
        transactions = store["credit_transactions"].get(user_id, [])
        if created:
            _save_compatibility_store_unlocked(store)

    return {
        "balance": wallet["balance"],
        "recentTransactions": [
            {
                "createdAt": item.get("created_at"),
                "delta": parse_number(item.get("delta"), 0),
                "description": item.get("description") or "",
                "id": item.get("id"),
                "kind": item.get("kind") or "activity",
                "metadata": item.get("metadata") or {},
            }
            for item in reversed(transactions[-8:])
            if isinstance(item, dict)
        ],
        "totalPurchased": wallet["total_purchased"],
        "totalSpent": wallet["total_spent"],
        "trialClaimedAt": wallet["trial_claimed_at"],
        "trialCredits": wallet["trial_credits"],
    }


def spend_compatibility_credits(
    user_id: str,
    amount: int,
    description: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    with COMPATIBILITY_STORE_LOCK:
        store = _load_compatibility_store_unlocked()
        wallet, _ = _ensure_compatibility_wallet_unlocked(store, user_id)

        if amount <= 0:
            return wallet

        balance = parse_number(wallet.get("balance"), FREE_TRIAL_CREDITS)
        if balance < amount:
            raise api_error(
                402,
                f"You need {amount} credits for this action. Please add credits on the pricing page.",
            )

        timestamp = now_iso()
        wallet["balance"] = balance - amount
        wallet["total_spent"] = parse_number(wallet.get("total_spent"), 0) + amount
        wallet["updated_at"] = timestamp

        store["credit_wallets"][user_id] = wallet
        transactions = store["credit_transactions"].get(user_id, [])
        store["credit_transactions"][user_id] = (
            transactions if isinstance(transactions, list) else []
        )
        store["credit_transactions"][user_id].append(
            {
                "created_at": timestamp,
                "delta": -amount,
                "description": description,
                "id": f"compat-credit-{uuid4().hex[:12]}",
                "kind": "spend",
                "metadata": metadata or {},
            }
        )
        _save_compatibility_store_unlocked(store)
        return wallet


def add_compatibility_credits(
    user_id: str,
    amount: int,
    description: str,
    kind: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    with COMPATIBILITY_STORE_LOCK:
        store = _load_compatibility_store_unlocked()
        wallet, _ = _ensure_compatibility_wallet_unlocked(store, user_id)
        timestamp = now_iso()

        wallet["balance"] = parse_number(wallet.get("balance"), FREE_TRIAL_CREDITS) + amount
        wallet["total_purchased"] = parse_number(wallet.get("total_purchased"), 0) + amount
        wallet["updated_at"] = timestamp

        store["credit_wallets"][user_id] = wallet
        transactions = store["credit_transactions"].get(user_id, [])
        store["credit_transactions"][user_id] = (
            transactions if isinstance(transactions, list) else []
        )
        store["credit_transactions"][user_id].append(
            {
                "created_at": timestamp,
                "delta": amount,
                "description": description,
                "id": f"compat-credit-{uuid4().hex[:12]}",
                "kind": kind,
                "metadata": metadata or {},
            }
        )
        _save_compatibility_store_unlocked(store)
        return wallet


def build_compatibility_billing_order(
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = payload or {}
    razorpay_order_id = clean_text(payload.get("razorpay_order_id"), 120)
    return {
        "amount": parse_number(payload.get("amount"), 0),
        "created_at": payload.get("created_at"),
        "credits": parse_number(payload.get("credits"), 0),
        "currency": clean_text(payload.get("currency"), 12) or "INR",
        "id": payload.get("id") or f"compat-order-{razorpay_order_id or uuid4().hex[:12]}",
        "plan_id": clean_text(payload.get("plan_id"), 80),
        "provider": clean_text(payload.get("provider"), 40) or "razorpay",
        "razorpay_order_id": razorpay_order_id,
        "razorpay_payment_id": clean_text(payload.get("razorpay_payment_id"), 120),
        "receipt": clean_text(payload.get("receipt"), 120),
        "status": clean_text(payload.get("status"), 40) or "created",
        "updated_at": payload.get("updated_at"),
        "user_id": clean_text(payload.get("user_id"), 120),
        "verified_at": payload.get("verified_at"),
    }


def create_compatibility_billing_order(
    payload: dict[str, Any],
) -> dict[str, Any]:
    with COMPATIBILITY_STORE_LOCK:
        store = _load_compatibility_store_unlocked()
        order = build_compatibility_billing_order(payload)
        order_id = order.get("razorpay_order_id")
        if not order_id:
            raise api_error(400, "Missing Razorpay order id.")
        store["billing_orders"][order_id] = order
        _save_compatibility_store_unlocked(store)
        return order


def get_compatibility_billing_order(
    user_id: str,
    razorpay_order_id: str,
) -> dict[str, Any] | None:
    with COMPATIBILITY_STORE_LOCK:
        store = _load_compatibility_store_unlocked()
        raw_order = store["billing_orders"].get(razorpay_order_id)
        if not isinstance(raw_order, dict):
            return None
        order = build_compatibility_billing_order(raw_order)
        if order.get("user_id") != user_id:
            return None
        return order


def update_compatibility_billing_order(
    user_id: str,
    razorpay_order_id: str,
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    with COMPATIBILITY_STORE_LOCK:
        store = _load_compatibility_store_unlocked()
        raw_order = store["billing_orders"].get(razorpay_order_id)
        if not isinstance(raw_order, dict):
            return None

        current_order = build_compatibility_billing_order(raw_order)
        if current_order.get("user_id") != user_id:
            return None

        updated_order = build_compatibility_billing_order({**current_order, **payload})
        store["billing_orders"][razorpay_order_id] = updated_order
        _save_compatibility_store_unlocked(store)
        return updated_order


def build_compatibility_consultation_request(
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = payload or {}
    return {
        "company_name": clean_text(payload.get("company_name"), 160),
        "created_at": payload.get("created_at"),
        "email": clean_text(payload.get("email"), 160),
        "id": payload.get("id") or f"compat-consultation-{uuid4().hex[:12]}",
        "message": clean_text(payload.get("message"), 2000),
        "name": clean_text(payload.get("name"), 120),
        "source_page": clean_text(payload.get("source_page"), 200),
        "status": clean_text(payload.get("status"), 40) or "pending",
        "team_size": clean_text(payload.get("team_size"), 80),
        "updated_at": payload.get("updated_at"),
        "user_id": clean_text(payload.get("user_id"), 120),
    }


def create_compatibility_consultation_request(
    payload: dict[str, Any],
) -> dict[str, Any]:
    with COMPATIBILITY_STORE_LOCK:
        store = _load_compatibility_store_unlocked()
        consultation = build_compatibility_consultation_request(payload)
        store["consultation_requests"][consultation["id"]] = consultation
        _save_compatibility_store_unlocked(store)
        return consultation


def build_compatibility_contact_request(
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = payload or {}
    return {
        "budget": clean_text(payload.get("budget"), 80),
        "created_at": payload.get("created_at"),
        "email": clean_text(payload.get("email"), 160),
        "id": payload.get("id") or f"compat-contact-{uuid4().hex[:12]}",
        "message": clean_text(payload.get("message"), 4000),
        "name": clean_text(payload.get("name"), 120),
        "source_page": clean_text(payload.get("source_page"), 200),
        "status": clean_text(payload.get("status"), 40) or "new",
        "updated_at": payload.get("updated_at"),
        "user_id": clean_text(payload.get("user_id"), 120),
    }


def create_compatibility_contact_request(
    payload: dict[str, Any],
) -> dict[str, Any]:
    with COMPATIBILITY_STORE_LOCK:
        store = _load_compatibility_store_unlocked()
        contact_request = build_compatibility_contact_request(payload)
        store["contact_requests"][contact_request["id"]] = contact_request
        _save_compatibility_store_unlocked(store)
        return contact_request


def _build_compatibility_job(
    raw_job: dict[str, Any],
    applicant_count: int = 0,
) -> dict[str, Any]:
    title = clean_text(raw_job.get("title"), 140)
    location = clean_text(raw_job.get("location"), 120)
    return {
        **raw_job,
        "applicantCount": applicant_count,
        "company_name": raw_job.get("company_name") or "Hiring team",
        "employment_type": raw_job.get("employment_type") or "Full-time",
        "google_search_url": raw_job.get("google_search_url")
        or build_google_job_search_url(title, location),
        "linkedin_search_url": raw_job.get("linkedin_search_url")
        or build_linkedin_job_search_url(title, location),
        "topMatchScore": parse_number(raw_job.get("topMatchScore"), 0),
    }


def _list_compatibility_jobs_unlocked(
    store: dict[str, Any],
    recruiter_id: str | None = None,
    *,
    only_active: bool = False,
) -> list[dict[str, Any]]:
    applications = store["applications"]
    applicant_counts: dict[str, int] = {}
    for raw_application in applications.values():
        if not isinstance(raw_application, dict):
            continue
        job_id = raw_application.get("job_id")
        if not isinstance(job_id, str):
            continue
        applicant_counts[job_id] = applicant_counts.get(job_id, 0) + 1

    jobs: list[dict[str, Any]] = []
    for raw_job in store["jobs"].values():
        if not isinstance(raw_job, dict):
            continue
        if recruiter_id and raw_job.get("recruiter_id") != recruiter_id:
            continue
        if only_active and raw_job.get("status") != "active":
            continue
        job_id = raw_job.get("id")
        jobs.append(
            _build_compatibility_job(
                raw_job,
                applicant_counts.get(job_id, 0) if isinstance(job_id, str) else 0,
            )
        )

    return sorted(jobs, key=lambda item: item.get("created_at") or "", reverse=True)


def list_compatibility_jobs(
    recruiter_id: str | None = None,
    *,
    only_active: bool = False,
) -> list[dict[str, Any]]:
    with COMPATIBILITY_STORE_LOCK:
        store = _load_compatibility_store_unlocked()
        return _list_compatibility_jobs_unlocked(
            store,
            recruiter_id,
            only_active=only_active,
        )


def get_compatibility_job(job_id: str) -> dict[str, Any] | None:
    with COMPATIBILITY_STORE_LOCK:
        store = _load_compatibility_store_unlocked()
        raw_job = store["jobs"].get(job_id)
        if not isinstance(raw_job, dict):
            return None
        jobs = _list_compatibility_jobs_unlocked(store, raw_job.get("recruiter_id"))
        for job in jobs:
            if job.get("id") == job_id:
                return job
        return None


def create_compatibility_job(
    recruiter_id: str,
    payload: dict[str, Any],
    recruiter_profile: dict[str, Any],
    recruiter_name: str,
) -> dict[str, Any]:
    with COMPATIBILITY_STORE_LOCK:
        store = _load_compatibility_store_unlocked()
        timestamp = now_iso()
        job_id = f"compat-job-{uuid4().hex[:12]}"
        raw_job = {
            "category": payload.get("category") or "",
            "company_name": recruiter_profile.get("company_name")
            or recruiter_name
            or "Hiring team",
            "created_at": timestamp,
            "description": payload.get("description") or "",
            "employment_type": payload.get("employmentType") or "Full-time",
            "google_search_url": build_google_job_search_url(
                payload.get("title") or "",
                payload.get("location") or "",
            ),
            "id": job_id,
            "linkedin_search_url": build_linkedin_job_search_url(
                payload.get("title") or "",
                payload.get("location") or "",
            ),
            "location": payload.get("location") or "",
            "recruiter_id": recruiter_id,
            "salary_max": payload.get("salaryMax"),
            "salary_min": payload.get("salaryMin"),
            "skills": build_job_skills(payload),
            "status": payload.get("status") or "active",
            "title": payload.get("title") or "",
            "updated_at": timestamp,
        }
        store["jobs"][job_id] = raw_job
        _save_compatibility_store_unlocked(store)
        return _build_compatibility_job(raw_job)


def create_compatibility_application(
    candidate_id: str,
    job_id: str,
    resume_id: str | None,
) -> tuple[dict[str, Any], bool]:
    with COMPATIBILITY_STORE_LOCK:
        store = _load_compatibility_store_unlocked()
        jobs = store["jobs"]
        raw_job = jobs.get(job_id)
        if not isinstance(raw_job, dict) or raw_job.get("status") != "active":
            raise api_error(404, "Job not found.")

        applications = store["applications"]
        for raw_application in applications.values():
            if not isinstance(raw_application, dict):
                continue
            if (
                raw_application.get("candidate_id") == candidate_id
                and raw_application.get("job_id") == job_id
            ):
                return raw_application, False

        timestamp = now_iso()
        application = {
            "applied_at": timestamp,
            "candidate_id": candidate_id,
            "id": f"compat-application-{uuid4().hex[:12]}",
            "job_id": job_id,
            "resume_id": resume_id,
            "status": "submitted",
            "updated_at": timestamp,
        }
        applications[application["id"]] = application
        _save_compatibility_store_unlocked(store)
        return application, True


def get_compatibility_candidate_applications(candidate_id: str) -> list[dict[str, Any]]:
    with COMPATIBILITY_STORE_LOCK:
        store = _load_compatibility_store_unlocked()
        jobs = store["jobs"]
        applications = []
        for raw_application in store["applications"].values():
            if not isinstance(raw_application, dict):
                continue
            if raw_application.get("candidate_id") != candidate_id:
                continue
            raw_job = jobs.get(raw_application.get("job_id"), {})
            if not isinstance(raw_job, dict):
                raw_job = {}
            applications.append(
                {
                    **raw_application,
                    "employmentType": raw_job.get("employment_type") or "Full-time",
                    "location": raw_job.get("location") or "Not specified",
                    "title": raw_job.get("title") or "Open role",
                }
            )

        return sorted(
            applications,
            key=lambda item: item.get("applied_at") or "",
            reverse=True,
        )


def get_compatibility_recruiter_dashboard(recruiter_id: str) -> dict[str, Any]:
    with COMPATIBILITY_STORE_LOCK:
        store = _load_compatibility_store_unlocked()
        jobs = _list_compatibility_jobs_unlocked(store, recruiter_id)
        recent_jobs = _list_compatibility_jobs_unlocked(store, only_active=True)
        job_ids = {job.get("id") for job in jobs}
        applications = [
            raw_application
            for raw_application in store["applications"].values()
            if isinstance(raw_application, dict)
            and raw_application.get("job_id") in job_ids
        ]

    return {
        "jobs": jobs,
        "recentJobs": recent_jobs,
        "shortlist": [],
        "stats": {
            "interviews": len(
                [item for item in applications if item.get("status") == "interview"]
            ),
            "newCandidates": len({item.get("candidate_id") for item in applications}),
            "openRoles": len([job for job in jobs if job.get("status") != "closed"]),
        },
    }


def build_compatibility_candidate_profile(
    user: dict[str, Any], role: str
) -> dict[str, Any]:
    stored = ((user.get("user_metadata") or {}).get(AI_HIRING_PROFILE_KEY) or {})
    latest_resume = ((user.get("user_metadata") or {}).get(AI_HIRING_RESUME_KEY) or {})
    return {
        "__compatibility_mode": True,
        "bio": stored.get("bio") or "",
        "headline": stored.get("headline") or "",
        "latest_resume_id": latest_resume.get("id"),
        "location": stored.get("location") or "",
        "profile_completion": stored.get("profile_completion")
        or compute_profile_completion(
            {
                "bio": stored.get("bio"),
                "headline": stored.get("headline"),
                "latestResumeId": latest_resume.get("id"),
                "location": stored.get("location"),
                "name": display_name(user),
                "skills": stored.get("skills") or [],
                "yearsExperience": stored.get("years_experience") or 0,
            }
        ),
        "role": role,
        "skills": stored.get("skills") or [],
        "user_id": user.get("id"),
        "years_experience": stored.get("years_experience") or 0,
    }


def build_compatibility_recruiter_profile(
    user: dict[str, Any], role: str
) -> dict[str, Any]:
    stored = ((user.get("user_metadata") or {}).get(AI_HIRING_PROFILE_KEY) or {})
    return {
        "__compatibility_mode": True,
        "company_name": stored.get("company_name") or "",
        "company_size": stored.get("company_size") or "",
        "industry": stored.get("industry") or "",
        "role": role,
        "user_id": user.get("id"),
        "website": stored.get("website") or "",
    }


def build_compatibility_resume_bundle(user: dict[str, Any] | None) -> dict[str, Any]:
    metadata = (user or {}).get("user_metadata") or {}
    stored = metadata.get(AI_HIRING_RESUME_KEY) or {}
    latest_resume = None
    parsing_result = None

    if stored.get("file_name"):
        latest_resume = {
            "file_name": stored.get("file_name"),
            "id": stored.get("id"),
            "parsing_status": stored.get("parsing_status") or "completed",
            "uploaded_at": stored.get("uploaded_at"),
        }
        parsing_result = {
            "skills": stored.get("skills") or [],
            "suggestions": stored.get("suggestions") or [],
            "summary": stored.get("summary") or "",
        }

    return {
        "history": [latest_resume] if latest_resume else [],
        "latestResume": latest_resume,
        "parsingResult": parsing_result,
    }


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


def parse_consultation_request_payload(
    payload: dict[str, Any],
    user: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = (user or {}).get("user_metadata") or {}
    fallback_name = clean_text(
        metadata.get("full_name") or metadata.get("name") or "",
        120,
    )
    fallback_email = clean_text((user or {}).get("email"), 160)

    name = clean_text(payload.get("name"), 120) or fallback_name
    email = clean_text(payload.get("email"), 160) or fallback_email
    if not name:
        raise api_error(400, "Name is required.")
    if "@" not in email or "." not in email.split("@")[-1]:
        raise api_error(400, "A valid work email is required.")

    return {
        "companyName": clean_text(payload.get("companyName"), 160),
        "email": email,
        "message": clean_text(payload.get("message"), 2000),
        "name": name,
        "sourcePage": clean_text(payload.get("sourcePage"), 200),
        "teamSize": clean_text(payload.get("teamSize"), 80),
    }


def parse_contact_request_payload(
    payload: dict[str, Any],
    user: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = (user or {}).get("user_metadata") or {}
    fallback_name = clean_text(
        metadata.get("full_name") or metadata.get("name") or "",
        120,
    )
    fallback_email = clean_text((user or {}).get("email"), 160)

    name = clean_text(payload.get("name"), 120) or fallback_name
    email = clean_text(payload.get("email"), 160) or fallback_email
    message = clean_text(payload.get("message"), 4000)

    if not name:
        raise api_error(400, "Name is required.")
    if "@" not in email or "." not in email.split("@")[-1]:
        raise api_error(400, "A valid email is required.")
    if not message:
        raise api_error(400, "Project details are required.")

    return {
        "budget": clean_text(payload.get("budget"), 80),
        "email": email,
        "message": message,
        "name": name,
        "sourcePage": clean_text(payload.get("sourcePage"), 200),
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
