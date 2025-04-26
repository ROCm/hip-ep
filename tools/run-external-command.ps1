function Run {
    param (
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Args  # Remaining arguments for the command
    )

    Write-Host "[running]:  $($Args -join ' ')"
    # Run the command using the call operator (&)
    &  $Args[0] $Args[1..$Args.Length]
    # Check the exit code
    if ($LASTEXITCODE -ne 0) {
        Write-Error "[fail] '$($Args -join ' ')' failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
    Write-Host "[ok]     :  $($Args -join ' ')"
}
