"""
groq_summary.py

Generates the AI clinical-style summary paragraph for the PDF report using
your existing Groq client/model — same pattern as your /chat endpoint.

Drop this next to your app.py (or import groq_client from app.py directly
instead of re-initializing it — see note at bottom of file).
"""

from groq import Groq

# Match your /chat endpoint's model exactly.
GROQ_SUMMARY_MODEL = "llama-3.3-70b-versatile"

SUMMARY_SYSTEM_PROMPT = (
    "You are a medical AI assistant that writes short, factual report "
    "summaries for an automated leukemia detection system. "
    "You write ONE paragraph (4-6 sentences) summarizing a single image "
    "analysis result for inclusion in a PDF report that clinicians will "
    "review. "
    "Rules: "
    "1) State the predicted class and confidence score plainly. "
    "2) If abnormal_region_count > 0, mention that the Grad-CAM heatmap "
    "highlighted that many region(s) of high model attention, without "
    "claiming this proves a diagnosis. "
    "3) If confidence is below 60%%, explicitly state the result is "
    "low-confidence and should be treated as inconclusive pending further "
    "review. "
    "4) NEVER state or imply a definitive diagnosis. Use hedged language "
    "like 'the model suggests' or 'is consistent with'. "
    "5) End by stating this is an AI-generated summary requiring "
    "confirmation by a qualified hematologist/oncologist. "
    "6) Do not use markdown, bullet points, or headers \u2014 plain prose only. "
    "7) Do not exceed 6 sentences."
)


def build_user_prompt(
    prediction_label: str,
    confidence: float,
    abnormal_region_count: int,
) -> str:
    confidence_pct = confidence * 100 if confidence <= 1 else confidence
    return (
        f"Predicted class: {prediction_label}\n"
        f"Confidence score: {confidence_pct:.1f}%\n"
        f"Grad-CAM abnormal region count: {abnormal_region_count}\n\n"
        f"Write the report summary paragraph now."
    )


def generate_ai_summary(
    groq_client: Groq,
    prediction_label: str,
    confidence: float,
    abnormal_region_count: int = 0,
    fallback_on_error: bool = True,
) -> str:
    """
    Calls Groq to write the report's AI-generated summary paragraph.

    Args:
        groq_client: your already-initialized Groq client (same one used
                     in /chat).
        prediction_label: e.g. "Acute Lymphoblastic Leukemia (ALL)"
        confidence: 0-1 float (e.g. 0.93) or 0-100 (e.g. 93.0) — handled
                    either way.
        abnormal_region_count: from analyze_heatmap_regions(); pass 0 if
                    you don't have it.
        fallback_on_error: if True, returns a safe templated sentence
                    instead of raising when Groq fails (recommended for
                    a report-generation endpoint — don't let a PDF fail
                    because of an LLM API hiccup).

    Returns:
        A short plain-text paragraph (str).
    """
    if groq_client is None:
        if fallback_on_error:
            return _fallback_summary(prediction_label, confidence, abnormal_region_count)
        raise RuntimeError("Groq client is not configured (GROQ_API_KEY missing).")

    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_SUMMARY_MODEL,
            messages=[
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_user_prompt(
                        prediction_label, confidence, abnormal_region_count
                    ),
                },
            ],
            max_tokens=300,
            temperature=0.3,  # low temperature: keep it factual/consistent for a clinical report
        )
        text = completion.choices[0].message.content.strip()
        return text if text else _fallback_summary(prediction_label, confidence, abnormal_region_count)

    except Exception as e:
        print(f"⚠️ Groq summary generation failed: {e}")
        if fallback_on_error:
            return _fallback_summary(prediction_label, confidence, abnormal_region_count)
        raise


def _fallback_summary(prediction_label: str, confidence: float, abnormal_region_count: int) -> str:
    """Safe templated fallback used only if the Groq call fails — keeps the
    report-generation endpoint from breaking due to an external API issue."""
    confidence_pct = confidence * 100 if confidence <= 1 else confidence

    text = (
        f"The model analyzed the uploaded blood smear image and classified it as "
        f"\u201c{prediction_label}\u201d with a confidence score of {confidence_pct:.1f}%. "
    )
    if abnormal_region_count > 0:
        text += (
            f"The Grad-CAM heatmap identified {abnormal_region_count} region(s) of "
            f"elevated model attention, which may be consistent with this prediction. "
        )
    if confidence_pct < 60:
        text += (
            "This confidence score is relatively low, and the result should be "
            "treated as inconclusive pending further clinical evaluation. "
        )
    text += (
        "This is an automatically generated summary and must be reviewed and "
        "confirmed by a qualified hematologist or oncologist before any "
        "clinical decision is made."
    )
    return text


# ---------------------------------------------------------------------------
# NOTE on reusing your existing groq_client instead of creating a new one:
#
# In your app.py you already have:
#     groq_client = Groq(api_key=GROQ_API_KEY)
#
# Don't re-initialize a second client in this file. Instead, in your /report
# route, just pass your existing `groq_client` variable straight into
# generate_ai_summary(groq_client=groq_client, ...). That keeps a single
# Groq client instance for the whole app.
# ---------------------------------------------------------------------------