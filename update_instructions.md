Claude has been running in a loop using ./run_agent.sh for some time. You can see the budget spent and
the ideas that have been tried in BUDGET.md and ides/README.md. While there has been some progress,
it has stalled and the agent seems increasingly stuck on incremental experiments that don't go anywhere.
Based on what has been attempted so far, I want you to re-work the agent prompt (and any other necessary files) to try and avoid past problems, and to make the experimentation more bold. Almost $500 has been spent on
little optimizations, I'd instead rather see some big bets investigated to see if we can make real progress.
Leave the history as is so that the running agent can learn from the past, but modify things to prompt 
the agent to be explicitly more bold.

One other issue I've noticed is too much time spent on local testing. Local tests are fine, but not actually "free" (the machine costs about $3/hr and we only have so much time). So I want the general format of local testing first to hold, but I'd rather the agent runs shorter local tests and then kicks up to the H100s when there is something promising. This is especially true given that local testing doesn't seem to have done a good job predicting real results.

Two things I've noticed in the last few days I'd like the agent to consider (and reject immediately if they don't make sense but try otherwise): https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/ and https://dao-lab.ai/blog/2026/gram-newton-schulz/ These may only be relevant to bigger models, please review them and add them as things to try if they could be relevant here.
