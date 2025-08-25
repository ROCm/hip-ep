##
## Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##
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
        Write-Error " $(Get-Date) [fail] '$($Args -join ' ')' failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
    Write-Host "$(Get-Date) [ok]     :  $($Args -join ' ')"
}
