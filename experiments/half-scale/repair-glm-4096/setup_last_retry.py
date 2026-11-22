"""Prepare a bounded recovery of the remaining failed GLM prompts."""

from pathlib import Path

folder = Path.cwd()
source = (folder / "retry_remaining.py").read_text()
changes = {
    "OUTPUT_LIMIT = 8192": "OUTPUT_LIMIT = 32768",
    '"max_concurrency": 40': '"max_concurrency": 6',
    '"retry-plan.json"': '"last-retry-plan.json"',
    "initial=24, maximum=40": "initial=4, maximum=6",
    '"timeout=240"': '"timeout=420"',
    """        reply = invoke(prompt)
        if reply.finish_reason == "max_tokens" or not reply.text.strip():
            reply = retry(prompt)
        if reply.finish_reason == "max_tokens" or not reply.text.strip():
            raise model_adapters.ModelError("GLM still incomplete at 16384 output tokens")""":
    """        try:
            reply = invoke(prompt)
        except model_adapters.ModelError as error:
            if str(error) != "GLM HTTP 400":
                raise
            reply = retry(prompt)
        if reply.finish_reason == "max_tokens" or not reply.text.strip():
            raise model_adapters.ModelError("GLM still incomplete with extended output limit")""",
}
for old,new in changes.items():
    assert source.count(old)==1, old
    source=source.replace(old,new,1)
compile(source, "retry_last.py", "exec")
(folder/"retry_last.py").write_text(source)
print("Residual retry runner compiled.")
