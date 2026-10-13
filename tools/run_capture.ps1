<#
Run the workspace MaixPy source in RAM over classic MicroPython raw REPL.
No flashing, reset, boot-file change, third-party package or SD write.
For the continuous basic demo, TimeoutSeconds ends capture with Ctrl-C.
Disconnect MaixPy IDE from the same port before running. Default output is a
new captures/*.txt file; existing logs are never overwritten. -SelfTest uses
synthetic protocol fragments and does not open a serial port.
#>
[CmdletBinding()]
param(
    [ValidatePattern('^COM[0-9]+$')][string]$Port = 'COM3',
    [ValidateRange(1, 4000000)][int]$Baud = 115200,
    [string]$ScriptPath = '',
    [string]$LogPath = '',
    [ValidateRange(10, 3600)][int]$TimeoutSeconds = 180,
    [switch]$SelfTest
)
$ErrorActionPreference = 'Stop'

function New-RawState {
    return [pscustomobject]@{ Phase = 'ack'; Buffer = ''; ErrorText = ''; Complete = $false }
}

function Add-RawChunk($State, [string]$Chunk, [scriptblock]$OnOutput) {
    $State.Buffer += $Chunk
    while (-not $State.Complete) {
        if ($State.Phase -eq 'ack') {
            if ($State.Buffer.Length -lt 2) { return }
            if (-not $State.Buffer.StartsWith('OK')) { throw 'Board did not acknowledge the source with OK.' }
            $State.Buffer = $State.Buffer.Substring(2)
            $State.Phase = 'stdout'
        }
        $marker = $State.Buffer.IndexOf([char]4)
        $part = if ($marker -lt 0) { $State.Buffer } else { $State.Buffer.Substring(0, $marker) }
        if ($State.Phase -eq 'stdout') { & $OnOutput $part }
        else { $State.ErrorText += $part }
        if ($marker -lt 0) { $State.Buffer = ''; return }
        $State.Buffer = $State.Buffer.Substring($marker + 1)
        if ($State.Phase -eq 'stdout') { $State.Phase = 'stderr' }
        else { $State.Complete = $true }
    }
}

if ($SelfTest) {
    foreach ($pieces in @(
        @('O', "K# one`n# two`n", [string][char]4, [string][char]4, '>'),
        @("OK# one`n# two`n$([char]4)$([char]4)>"),
        @('OK', "# one`n", "# two`n$([char]4)", "error$([char]4)>"))) {
        $state = New-RawState
        $collected = New-Object System.Text.StringBuilder
        foreach ($piece in $pieces) { Add-RawChunk $state $piece { param($s) [void]$collected.Append($s) } }
        if (-not $state.Complete -or $collected.ToString() -ne "# one`n# two`n") { throw 'Fragmented raw REPL output self-test failed.' }
    }
    if ($state.ErrorText -ne 'error') { throw 'Raw REPL stderr self-test failed.' }
    $state = New-RawState
    try { Add-RawChunk $state 'NO' { param($s) }; throw 'Invalid ACK was accepted.' }
    catch { if ($_.Exception.Message -eq 'Invalid ACK was accepted.') { throw } }
    Write-Output 'Raw REPL parser self-tests passed; no hardware used.'
    exit 0
}

$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $ScriptPath) { $ScriptPath = Join-Path $projectRoot 'k210_rppg_basic.py' }
$codePath = (Resolve-Path -LiteralPath $ScriptPath).Path
$sourceBytes = [System.IO.File]::ReadAllBytes($codePath)
if (-not $LogPath) {
    $captureDir = Join-Path $projectRoot 'captures'
    [void][System.IO.Directory]::CreateDirectory($captureDir)
    $LogPath = Join-Path $captureDir ('k210_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff') + '.txt')
}
$logAbsolute = [System.IO.Path]::GetFullPath($LogPath)
$serial = New-Object System.IO.Ports.SerialPort $Port, $Baud, None, 8, One
$serial.DtrEnable = $false
$serial.RtsEnable = $false
$serial.Handshake = [System.IO.Ports.Handshake]::None
$serial.Encoding = New-Object System.Text.UTF8Encoding $false
$serial.ReadTimeout = 200
$serial.WriteTimeout = 5000
$writer = $null
try {
    try { $serial.Open() }
    catch { throw "Cannot open $Port. Disconnect MaixPy IDE/other serial clients first. $($_.Exception.Message)" }
    $fileStream = [System.IO.FileStream]::new($logAbsolute, [System.IO.FileMode]::CreateNew,
        [System.IO.FileAccess]::Write, [System.IO.FileShare]::Read)
    $writer = [System.IO.StreamWriter]::new($fileStream, [System.Text.UTF8Encoding]::new($false))
    $writer.AutoFlush = $true
    $writer.WriteLine("# host_port=$Port; baud=$Baud; source=$codePath")
    Write-Host "Capturing to $logAbsolute"
    # Interrupt the previous RAM program, enter raw REPL. Do not soft-reset.
    $serial.Write([string][char]13 + [char]3 + [char]3)
    Start-Sleep -Milliseconds 250
    $serial.DiscardInBuffer()
    $serial.Write([string][char]13 + [char]1)
    $banner = ''
    $until = [DateTime]::UtcNow.AddSeconds(10)
    while ([DateTime]::UtcNow -lt $until) {
        $banner += $serial.ReadExisting()
        if ($banner.Contains('raw REPL') -and $banner.EndsWith('>')) { break }
        Start-Sleep -Milliseconds 20
    }
    if (-not ($banner.Contains('raw REPL') -and $banner.EndsWith('>'))) {
        throw 'No classic raw REPL prompt. No new source was executed. Reconnect/reset manually and retry.'
    }
    for ($offset = 0; $offset -lt $sourceBytes.Length; $offset += 256) {
        $serial.Write($sourceBytes, $offset, [Math]::Min(256, $sourceBytes.Length - $offset))
        Start-Sleep -Milliseconds 20
    }
    $serial.Write([string][char]4)
    $state = New-RawState
    $consoleLine = New-Object System.Text.StringBuilder
    $consume = {
        param([string]$text)
        $writer.Write($text)
        [void]$consoleLine.Append($text)
        $current = $consoleLine.ToString()
        $newline = $current.IndexOf("`n")
        while ($newline -ge 0) {
            $line = $current.Substring(0, $newline).TrimEnd("`r")
            # Save ALL stdout, but show only summaries/results on the host console.
            if ($line.StartsWith('#')) { Write-Host $line }
            $current = $current.Substring($newline + 1)
            $newline = $current.IndexOf("`n")
        }
        [void]$consoleLine.Clear()
        [void]$consoleLine.Append($current)
    }
    $until = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    while (-not $state.Complete -and [DateTime]::UtcNow -lt $until) {
        $chunk = $serial.ReadExisting()
        if ($chunk.Length) { Add-RawChunk $state $chunk $consume }
        else { Start-Sleep -Milliseconds 10 }
    }
    if (-not $state.Complete) {
        $writer.WriteLine("`n# host_capture_duration_reached; sending_interrupt=1")
        $serial.Write([string][char]3)
        $stopDeadline = [DateTime]::UtcNow.AddSeconds(10)
        while (-not $state.Complete -and [DateTime]::UtcNow -lt $stopDeadline) {
            $chunk = $serial.ReadExisting()
            if ($chunk.Length) { Add-RawChunk $state $chunk $consume }
            else { Start-Sleep -Milliseconds 10 }
        }
        if (-not $state.Complete) {
            $writer.WriteLine("`n# host_timeout; board_stop_unconfirmed=1")
            throw "Board did not finish after Ctrl-C. Output is saved in $logAbsolute; stop it in the IDE."
        }
    }
    if ($state.ErrorText.Length) {
        $writer.WriteLine("`n# host_board_error")
        $writer.Write($state.ErrorText)
        throw "Board reported an exception: $($state.ErrorText)"
    }
    $serial.Write([string][char]2) # Return to friendly REPL after program completion.
    Write-Host "Board completed. Full log saved: $logAbsolute"
} finally {
    if ($null -ne $writer) { $writer.Dispose() }
    if ($serial.IsOpen) { $serial.Close() }
    $serial.Dispose()
}
