"""
Seeds a 50-message session across 5 topics for the semantic retrieval eval.

Usage:
    python evals/seed.py [--base-url http://localhost:8081] [--clean]

Topics:
    msgs 1–10  : Italian restaurant discussion
    msgs 11–20 : Japan trip planning
    msgs 21–30 : Work deadline stress
    msgs 31–40 : Movie recommendation (Dune Part Two)
    msgs 41–50 : Headache symptoms and remedies
"""

import argparse
import asyncio
import httpx

AGENT_ID = "eval-agent"
SESSION_ID = "eval-session-50"

MESSAGES = [
    # ── Topic A: Italian restaurant (msgs 1–10) ──────────────────────────────
    ("user",      "Hey, I visited that new Italian place on 5th Avenue yesterday."),
    ("assistant", "Oh nice! What did you think of it?"),
    ("user",      "The pasta carbonara was incredible, best I've had in years."),
    ("assistant", "That's a strong endorsement. Did you make a reservation or walk in?"),
    ("user",      "Made a reservation for two weeks out, it's really popular."),
    ("assistant", "Two weeks out — must be worth the wait. What was the vibe like?"),
    ("user",      "Very cozy, dim lighting, authentic Italian décor. The owner is from Naples."),
    ("assistant", "A Neapolitan-run restaurant — that explains the quality."),
    ("user",      "Exactly. They also do a fantastic tiramisu if you're into desserts."),
    ("assistant", "Noted. I'll keep it in mind for a special occasion."),

    # ── Topic B: Japan trip planning (msgs 11–20) ─────────────────────────────
    ("user",      "Totally different topic — I'm planning a trip to Japan next spring."),
    ("assistant", "Exciting! Tokyo or are you going elsewhere too?"),
    ("user",      "Tokyo for sure, but I want to spend a few days in Kyoto too."),
    ("assistant", "Classic combo. Cherry blossom season in Kyoto is spectacular in late March."),
    ("user",      "That's exactly when I'm thinking — late March to early April."),
    ("assistant", "Flights from the US to Tokyo are cheapest mid-week. Have you looked yet?"),
    ("user",      "I found a layover in Seoul that actually makes it pretty affordable."),
    ("assistant", "Seoul layovers are great — even a few hours lets you try Korean street food."),
    ("user",      "Good point. I'm also considering a ryokan stay in Kyoto for one night."),
    ("assistant", "A ryokan is worth it for the experience — book well in advance though."),

    # ── Topic C: Work deadline stress (msgs 21–30) ────────────────────────────
    ("user",      "Ugh, work has been really stressful lately."),
    ("assistant", "Sorry to hear that. What's going on?"),
    ("user",      "We have a Q3 report due at the end of the month and I'm behind."),
    ("assistant", "That kind of deadline pressure is rough. How far behind are you?"),
    ("user",      "About three days of work behind. My manager keeps pinging me for updates."),
    ("assistant", "That's stressful. Is there anything you can cut from scope to catch up?"),
    ("user",      "Maybe. The appendix section feels like overkill for this audience."),
    ("assistant", "Dropping the appendix sounds reasonable — focus on what actually gets read."),
    ("user",      "Yeah I think I'll do that. Just needed to hear it from someone else."),
    ("assistant", "Sometimes you just need permission to simplify. You've got this."),

    # ── Topic D: Movie recommendation (msgs 31–40) ────────────────────────────
    ("user",      "Have you heard much about Dune Part Two? Thinking of watching it."),
    ("assistant", "It's getting great reviews — the visual effects are supposed to be stunning."),
    ("user",      "I loved the first one. The sandworm scenes were wild."),
    ("assistant", "Part Two apparently goes even bigger on the scale. Zendaya has a much larger role."),
    ("user",      "Oh interesting, she was barely in the first one."),
    ("assistant", "Exactly. Her character becomes central to the second half of the book's story."),
    ("user",      "Good to know. Should I rewatch Part One first to refresh myself?"),
    ("assistant", "Probably worth it — there are a lot of characters and faction names to track."),
    ("user",      "Fair point. I'll do a double feature this weekend then."),
    ("assistant", "Perfect plan. Let me know what you think after!"),

    # ── Topic E: Headache symptoms (msgs 41–50) ───────────────────────────────
    ("user",      "Random question — I've been getting these persistent headaches lately."),
    ("assistant", "Sorry to hear that. How long have they been going on?"),
    ("user",      "About a week. Usually behind my eyes, worse in the afternoon."),
    ("assistant", "That pattern can be tension headaches or eye strain from screens."),
    ("user",      "I have been staring at my monitor a lot more since the deadline crunch."),
    ("assistant", "Screen fatigue is a likely culprit. Have you tried the 20-20-20 rule?"),
    ("user",      "No, what's that?"),
    ("assistant", "Every 20 minutes, look at something 20 feet away for 20 seconds. Reduces eye strain."),
    ("user",      "That's simple enough. Any other remedies I should try?"),
    ("assistant", "Stay hydrated, reduce caffeine in the afternoon, and make sure your monitor isn't too bright."),
]


async def seed(base_url: str, clean: bool) -> None:
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        if clean:
            resp = await client.delete(f"/memory/{AGENT_ID}/{SESSION_ID}")
            if resp.status_code in (200, 404):
                print(f"Cleaned existing session {SESSION_ID}")
            else:
                print(f"Warning: DELETE returned {resp.status_code}")

        print(f"Seeding {len(MESSAGES)} messages to {AGENT_ID}/{SESSION_ID} ...")
        for i, (role, content) in enumerate(MESSAGES, 1):
            resp = await client.post(
                f"/memory/{AGENT_ID}/{SESSION_ID}/append",
                json={"role": role, "content": content},
            )
            resp.raise_for_status()
            if i % 10 == 0:
                print(f"  {i}/{len(MESSAGES)} messages written")

        print(f"\nDone. Session seeded with {len(MESSAGES)} messages.")
        print(f"Topics: Italian restaurant (1-10), Japan trip (11-20), "
              f"Work stress (21-30), Dune movie (31-40), Headaches (41-50)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8081")
    parser.add_argument("--clean", action="store_true", help="Delete session before seeding")
    args = parser.parse_args()
    asyncio.run(seed(args.base_url, args.clean))
