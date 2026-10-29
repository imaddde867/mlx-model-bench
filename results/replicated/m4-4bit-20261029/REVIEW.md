# Review of the 2026-10-29 replicated study

**Condition-flagged exploratory evidence, not a verified quiet-machine run.**
Read this alongside [the generated report](REPLICATED_STUDY.md).

Independent validation: 20/20 sessions; all 360 raw responses (300 kept plus
60 discarded warmups) stopped normally and matched the duration target.
All rows recorded AC power, low-power mode off and the frozen common cap 512.
The unattended batch exited 0 at 2026-10-29 05:11:43 UTC (07:11 Helsinki).

All 20 sessions have observed caveats, including substantial background CPU
interference, 2.313–4.077 GB swap in kept-trial snapshots, and unavailable
CPU thermal-speed telemetry. Pre-block quiet checks and cooldowns did not
prevent activity during sessions. No compromised sessions were dropped or
relabeled. Intent `clean` does not establish interference-free conditions.

LFM generates 153/189/290 tokens for short/medium/long, including reasoning;
the other models generate four tokens. Its TTFT is the first generated reasoning
token, not time to the visible final answer. Comparisons describe this fixed
retrieval workload's response completion, not equal-output decode performance.
See the generated report for each checkpoint's publisher/bpw, tokenizer-specific
prompt lengths, actual output lengths, ranges and exploratory session intervals.
Five sessions and pilot-informed prompts do not establish population quality,
agent capability, universal ranking or stable confidence-interval coverage.

Raw evidence and the frozen plan are preserved unchanged. Historical P2/P3
records remain separate. A stronger quiet-machine claim needs a new separately
planned run; this run should not be silently replaced or pooled with one.
