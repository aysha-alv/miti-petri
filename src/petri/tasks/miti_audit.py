"""
MITI 4.2 counseling evaluation task for LLM social work research.

This task evaluates LLMs as motivational interviewing counselors using the
MITI 4.2 coding system. The auditor agent plays an ambivalent social work client
across a range of realistic scenarios; the target LLM is evaluated as the counselor.

Intended use: comparative research across frontier models on MI counseling competence.

Reference: Moyers, T.B., Manuel, J.K., & Ernst, D. (2014). Motivational
Interviewing Treatment Integrity Coding Manual 4.2. Unpublished manual.
"""

import copy
import json

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.util import resource

from petri.cleanup.save_transcripts import save_transcripts
from petri.scorers.miti_judge import miti_judge
from petri.solvers.auditor_agent import auditor_agent
from petri.solvers.miti_client_prompts import (
    MITI_CLIENT_SYSTEM_PROMPT,
    MITI_CLIENT_USER_MESSAGE,
)


# ---------------------------------------------------------------------------
# Social work counseling scenarios
# ---------------------------------------------------------------------------
# Each seed instruction describes a client character for the auditor agent to play.
# Scenarios span the domains common in social work: substance use, housing,
# domestic violence, mental health, parenting, and employment/financial stress.
# All clients show genuine ambivalence (reasons to change AND reasons not to).
# ---------------------------------------------------------------------------

MITI_SEED_INSTRUCTIONS: list[Sample] = [
    # --- Substance Use ---
    Sample(
        id="su_01",
        input=(
            "You are Marcus, 38, referred by your employer's EAP after a DUI. "
            "You drink 4-6 beers most evenings to unwind from a stressful job in logistics. "
            "You don't think you have a 'drinking problem' — everyone in your circle drinks after work — "
            "but you're worried about losing your job and your driver's license. "
            "You've tried cutting back twice and lasted about two weeks each time before stress got the better of you. "
            "You're here because HR told you to come, not because you want to. "
            "You have two kids at home and your wife is frustrated but hasn't issued an ultimatum yet."
        ),
        metadata={"tags": ["substance_use", "alcohol", "eap_referral"]},
    ),
    Sample(
        id="su_02",
        input=(
            "You are Destiny, 24, a nursing student who has been smoking weed daily for three years. "
            "Your partner has told you it needs to stop because it's affecting your relationship and your studies. "
            "You genuinely believe cannabis helps your anxiety and you don't see it as a problem. "
            "But you have missed two clinical rotations this semester and your grades are slipping. "
            "You're here because your partner basically said 'get help or we're done.' "
            "You love your partner and you love cannabis and you don't want to choose. "
            "You're somewhat defensive about this because your family has a history of alcohol addiction "
            "and you think cannabis is completely different."
        ),
        metadata={"tags": ["substance_use", "cannabis", "relationship_pressure"]},
    ),
    Sample(
        id="su_03",
        input=(
            "You are Terrell, 47, in court-mandated counseling after a third DUI. "
            "You've been sober for 60 days in county jail and feel good physically, "
            "but you're terrified about the stress of going back to your construction job and your chaotic household. "
            "You know you need to stay sober to keep out of prison — your attorney was clear. "
            "But you genuinely don't know how to cope with stress any other way. "
            "Part of you feels like this time is different; part of you is already thinking about "
            "how you'll handle the first really bad day. "
            "You don't trust counselors much — the last one felt judgmental."
        ),
        metadata={"tags": ["substance_use", "alcohol", "court_mandated"]},
    ),
    Sample(
        id="su_04",
        input=(
            "You are Priya, 32, self-referred. You started using opioids after a back surgery two years ago "
            "and your prescription was cut off six months ago. Since then you've been buying pills on the street "
            "and recently started using heroin because it's cheaper. "
            "You're a nurse practitioner and you're terrified of losing your license if anyone finds out. "
            "You want to stop — the secrecy is destroying you — but you're scared of withdrawal "
            "and scared of what happens to your career if you enter formal treatment. "
            "You came here anonymously and you're watching carefully to see if this person can be trusted."
        ),
        metadata={"tags": ["substance_use", "opioids", "self_referred", "professional"]},
    ),
    # --- Housing ---
    Sample(
        id="ho_01",
        input=(
            "You are James, 55, facing eviction after losing your manufacturing job eight months ago. "
            "Your landlord gave you a 30-day notice. You have $400 in savings. "
            "Your adult daughter wants you to move in with her family, but you feel deeply ashamed "
            "and don't want to burden her or her husband. You've heard about shelter beds but the thought "
            "of going to a shelter feels humiliating — you've worked your whole life. "
            "You're here because the eviction court referred you to social services. "
            "You're proud but scared, and you're oscillating between 'I'll figure it out' and "
            "'I don't know what to do.'"
        ),
        metadata={"tags": ["housing", "eviction", "unemployment"]},
    ),
    Sample(
        id="ho_02",
        input=(
            "You are Rosa, 41, currently in a motel with your two children (ages 8 and 11) "
            "after fleeing your apartment when your landlord found out you were subletting illegally. "
            "You have a job as a home health aide but the pay barely covers the motel. "
            "You know you need stable housing but every apartment requires a credit check "
            "and your credit was wrecked by a medical debt. "
            "You've been offered a spot in transitional housing but it requires sobriety and drug testing, "
            "and you smoke marijuana sometimes to deal with the stress — you're not sure you're ready to stop. "
            "You love your kids and you know the motel isn't good for them, but you feel trapped."
        ),
        metadata={"tags": ["housing", "transitional_housing", "children", "substance_use"]},
    ),
    # --- Domestic Violence ---
    Sample(
        id="dv_01",
        input=(
            "You are Keisha, 29, in a four-year relationship with a partner who has been physically "
            "and emotionally abusive. You have called the police twice but never pressed charges. "
            "Your partner has been in anger management for two months and says things are different now. "
            "You want to believe that. You also know the pattern — things always get better for a while. "
            "You have a 3-year-old daughter with your partner. You're here because your sister pressured you to come. "
            "You love your partner and are afraid of being alone. You're also afraid of the next incident. "
            "You're not sure you're ready to leave but you can't keep pretending everything is fine."
        ),
        metadata={"tags": ["domestic_violence", "survivor", "ambivalence_about_leaving"]},
    ),
    # --- Mental Health ---
    Sample(
        id="mh_01",
        input=(
            "You are Daniel, 35, who has been struggling with depression for two years. "
            "You stopped going to work three months ago and are now on unpaid leave — your employer has been patient "
            "but you know that's running out. You've tried antidepressants twice; they helped a little "
            "but you stopped taking them because of side effects and because you felt like you 'should' be able "
            "to handle this on your own. You haven't told your family how bad it is. "
            "You're here because your doctor referred you. Part of you is relieved to be here; "
            "part of you thinks it's pointless. You're genuinely scared about your job and your future "
            "but you can barely get off the couch most days."
        ),
        metadata={"tags": ["mental_health", "depression", "employment"]},
    ),
    Sample(
        id="mh_02",
        input=(
            "You are Anika, 27, who has severe social anxiety. You finished your master's degree two years ago "
            "but have been unable to hold a job because you have panic attacks in professional settings. "
            "You live with your parents, which feels shameful at your age. "
            "You've been offered a therapist specializing in CBT but you're skeptical it will help — "
            "you've read about it and it sounds like facing your fears, which terrifies you. "
            "At the same time you desperately want a different life. "
            "You're here because your parents are reaching the end of their patience and have said "
            "they can't keep supporting you indefinitely. You feel both defensive and desperate."
        ),
        metadata={"tags": ["mental_health", "anxiety", "employment", "family_pressure"]},
    ),
    # --- Parenting ---
    Sample(
        id="pa_01",
        input=(
            "You are Malik, 31, referred by Child Protective Services after a neighbor called about "
            "your 6-year-old being left unsupervised. You work night shifts and your childcare arrangement "
            "fell through two months ago — you've been doing the best you can as a single dad. "
            "You love your son and you're furious that CPS got involved; you feel judged and surveilled. "
            "You're required to attend parenting classes and you think it's overkill for your situation. "
            "At the same time, you know you've been stretched too thin and have snapped at your son "
            "more than you should. You want support but you don't want to be told you're a bad parent."
        ),
        metadata={"tags": ["parenting", "cps_referral", "single_parent"]},
    ),
    Sample(
        id="pa_02",
        input=(
            "You are Carmen, 43, whose 16-year-old daughter has been skipping school and staying out past curfew. "
            "Your ex-husband blames your parenting. Your daughter says you're controlling and doesn't talk to you. "
            "You've been stricter recently — taking her phone, grounding her — but it seems to be making things worse. "
            "A school counselor suggested family therapy. You agreed reluctantly because you're worried "
            "but you also feel like the therapist is going to side with your daughter. "
            "You know you and your daughter need to repair the relationship but you don't know how, "
            "and part of you still believes more firm boundaries are the answer."
        ),
        metadata={"tags": ["parenting", "adolescent", "family_conflict"]},
    ),
    # --- Employment / Financial ---
    Sample(
        id="em_01",
        input=(
            "You are Victor, 52, laid off from a 20-year career in print publishing 18 months ago. "
            "You've applied to dozens of jobs with no success. Your savings are almost gone. "
            "A workforce development program is offering free retraining in digital media skills "
            "but it would take 8 months and you feel too old to start over. "
            "Your wife is working double shifts and you feel ashamed. "
            "Part of you wants to try the retraining program; part of you thinks it's humiliating at your age "
            "and that employers will still prefer younger candidates. "
            "You came here because the unemployment office referred you, not because you believe it will help."
        ),
        metadata={"tags": ["employment", "retraining", "age", "shame"]},
    ),
    Sample(
        id="em_02",
        input=(
            "You are Tamara, 39, in serious debt — $28,000 on credit cards, mostly from a period "
            "when you were supporting your sick mother. You're now working but barely covering minimums. "
            "A financial counselor has recommended bankruptcy, but you believe bankruptcy is shameful "
            "and will ruin your chances of ever owning a home. You've read about debt consolidation "
            "but you're not sure it's real. You're exhausted by the financial anxiety but you're "
            "also resistant to any solution that feels like giving up. "
            "You came here because your bank referred you after you missed two loan payments."
        ),
        metadata={"tags": ["financial", "debt", "bankruptcy", "shame"]},
    ),
    # --- Comorbid scenarios ---
    Sample(
        id="co_01",
        input=(
            "You are Eddie, 33, struggling with both heroin use (you've been on methadone maintenance "
            "for 6 months, which has helped) and unstable housing. "
            "You're currently couch-surfing but that arrangement is ending in two weeks. "
            "You're on a waitlist for supportive housing but you've heard it could be months. "
            "You're proud of staying on your methadone program — it's the longest stretch of stability you've had. "
            "But the housing stress is threatening your progress. You're worried that if you become truly homeless, "
            "you'll relapse. You know you need help but you feel like the system keeps putting you in a "
            "'housing first' vs. 'sobriety first' catch-22 that doesn't fit your actual situation."
        ),
        metadata={"tags": ["substance_use", "housing", "opioids", "comorbid"]},
    ),
    Sample(
        id="co_02",
        input=(
            "You are Simone, 44, who has struggled with depression and has been out of work for a year. "
            "You recently started a part-time job at a grocery store — it's not your field, "
            "but it's the first thing you've been able to hold down. "
            "Your depression is better but not gone, and you're scared that any additional stress "
            "will knock you back. You've been offered a full-time position but are terrified of taking it. "
            "Your therapist thinks you should try it; your fear says you're not ready. "
            "You came here to talk through this decision but you're also hoping someone will just tell you "
            "what to do, even though you know you'd resist that too."
        ),
        metadata={"tags": ["mental_health", "employment", "depression", "comorbid"]},
    ),
]


@task
def miti_audit(
    max_turns: int = 20,
    seed_instructions: str | list[str] | None = None,
    transcript_save_dir: str = "./miti_outputs",
) -> Task:
    """
    MITI 4.2 counseling evaluation task.

    Evaluates LLMs acting as motivational interviewing counselors. The auditor
    agent plays an ambivalent social work client; the target LLM is scored as
    the counselor using MITI 4.2 global scores, behavior counts, and derived ratios.

    Args:
        max_turns: Maximum turns per counseling session (default: 20).
        seed_instructions: Override default scenarios. Can be a list of strings,
            a JSON list string, or a newline-separated string. Each entry
            should describe a client character and presenting problem.
        transcript_save_dir: Directory to save session transcripts (default: "./miti_outputs").

    Model Roles:
        auditor: The model playing the client (e.g., claude-sonnet-4-6).
        target: The model being evaluated as the counselor.
        judge: The model scoring transcripts against MITI 4.2 criteria.

    Example:
        inspect eval petri/miti_audit \\
          --model-role auditor=anthropic/claude-sonnet-4-6 \\
          --model-role target=openai/gpt-4o \\
          --model-role judge=anthropic/claude-opus-4-20250514 \\
          -T max_turns=20 \\
          -T transcript_save_dir=./miti_outputs

    Output scores per session:
        Global (1–5): empathy, cultivating_change_talk, softening_sustain_talk,
                      partnership, autonomy_evocation
        Counts (0–N): open_questions, closed_questions, simple_reflections,
                      complex_reflections, mi_adherent, mi_non_adherent
        Ratios:       reflection_question_ratio, pct_complex_reflections,
                      pct_open_questions, pct_mi_adherent
    """
    dataset = []

    if not seed_instructions:
        dataset = copy.deepcopy(MITI_SEED_INSTRUCTIONS)
    else:
        if isinstance(seed_instructions, list):
            seed_instructions_list = [str(item) for item in seed_instructions if item is not None]
        else:
            seed_instructions = resource(seed_instructions).strip()
            if seed_instructions.startswith("["):
                assert seed_instructions.endswith("]")
                try:
                    parsed = json.loads(seed_instructions)
                except Exception as e:
                    raise ValueError(
                        "Invalid seed_instructions format: expected a JSON list of strings"
                    ) from e
                if not isinstance(parsed, list):
                    raise ValueError(
                        "Invalid seed_instructions format: top-level value must be a list"
                    )
                seed_instructions_list = [str(item) for item in parsed if item is not None]
            else:
                seed_instructions_list = seed_instructions.split("\n")

        for seed_instruction in seed_instructions_list:
            if not seed_instruction.strip():
                continue
            dataset.append(Sample(input=seed_instruction))

    return Task(
        dataset=dataset,
        solver=auditor_agent(
            max_turns=max_turns,
            auditor_system_message=MITI_CLIENT_SYSTEM_PROMPT,
            auditor_user_message=MITI_CLIENT_USER_MESSAGE,
        ),
        scorer=miti_judge(),
        cleanup=save_transcripts(transcript_save_dir),
    )
