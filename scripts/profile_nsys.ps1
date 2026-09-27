param(
    [string]$Output = "profiles/v41",
    [Parameter(Mandatory = $true, ValueFromRemainingArguments = $true)]
    [string[]]$Command
)

if (-not (Get-Command nsys -ErrorAction SilentlyContinue)) {
    throw "nsys is required on the Linux GPU host; this wrapper is for command documentation only."
}
if ($Command.Count -gt 0 -and $Command[0] -eq "--") {
    $Command = $Command[1..($Command.Count - 1)]
}
if ($Command.Count -eq 0) {
    throw "Usage: .\scripts\profile_nsys.ps1 -Output profiles\v41 -- command args"
}

$parent = Split-Path -Parent $Output
if ($parent) { New-Item -ItemType Directory -Force $parent | Out-Null }
& nsys profile --trace=cuda,nvtx,osrt --cuda-memory-usage=true `
    --force-overwrite=true -o $Output @Command
exit $LASTEXITCODE
