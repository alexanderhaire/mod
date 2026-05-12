# Vendor Quote Ingest — One-Time Setup

This runbook walks through the one-time work needed before `vendor_quote_ingest.py`
can pull mail from Outlook. After the seven steps below, the scheduled task
will run every 15 minutes unattended.

---

## 1. Generate a self-signed certificate (PowerShell, admin)

```powershell
$cert = New-SelfSignedCertificate `
    -Subject "CN=vendor-quote-ingest" `
    -CertStoreLocation "Cert:\CurrentUser\My" `
    -KeyExportPolicy Exportable `
    -KeySpec Signature `
    -KeyLength 2048 `
    -KeyAlgorithm RSA `
    -HashAlgorithm SHA256 `
    -NotAfter (Get-Date).AddYears(3)

$cert.Thumbprint    # copy — you'll need this
```

Export the public key for Azure upload:

```powershell
Export-Certificate -Cert $cert -FilePath "$HOME\vendor-quote-ingest.cer"
```

Export the private key (PEM, no password — protect the file with NTFS perms):

```powershell
$pwd = ConvertTo-SecureString -String "tmp" -Force -AsPlainText
Export-PfxCertificate -Cert $cert -FilePath "$HOME\vendor-quote-ingest.pfx" -Password $pwd
# convert to PEM
& "C:\Program Files\Git\usr\bin\openssl.exe" pkcs12 -in "$HOME\vendor-quote-ingest.pfx" -nodes -nocerts -out "$HOME\vendor-quote-ingest.key" -passin pass:tmp
& "C:\Program Files\Git\usr\bin\openssl.exe" pkcs12 -in "$HOME\vendor-quote-ingest.pfx" -nokeys -clcerts -out "$HOME\vendor-quote-ingest-cert.pem" -passin pass:tmp
Get-Content "$HOME\vendor-quote-ingest.key", "$HOME\vendor-quote-ingest-cert.pem" | Set-Content "$HOME\vendor-quote-ingest.pem"
Remove-Item "$HOME\vendor-quote-ingest.pfx", "$HOME\vendor-quote-ingest.key", "$HOME\vendor-quote-ingest-cert.pem"
```

The combined `vendor-quote-ingest.pem` is what `secrets.toml` will reference.

## 2. Register the Azure AD app

1. Go to https://entra.microsoft.com → **Identity** → **Applications** → **App registrations** → **New registration**.
2. Name: `Vendor Quote Ingest`. Account types: *Accounts in this organizational directory only*. Redirect URI: leave blank.
3. After creation, copy **Application (client) ID** and **Directory (tenant) ID** — you'll need both.

## 3. Upload the cert to the app registration

1. App registration → **Certificates & secrets** → **Certificates** → **Upload certificate**.
2. Upload `vendor-quote-ingest.cer` from step 1.
3. Confirm the thumbprint matches the one you captured.

## 4. Grant Mail.Read

1. App registration → **API permissions** → **Add a permission** → **Microsoft Graph** → **Application permissions** → check **Mail.Read** → **Add**.
2. Click **Grant admin consent for <tenant>**. Status should turn green.

## 5. (Recommended) Scope to one mailbox

By default, **Mail.Read** is tenant-wide. To scope it to only the procurement
mailbox, use `New-ApplicationAccessPolicy` from Exchange Online PowerShell:

```powershell
Connect-ExchangeOnline -UserPrincipalName admin@yourdomain.com

New-DistributionGroup -Name "VendorQuoteIngestScope" -Members "procurement@yourdomain.com" -Type Security
New-ApplicationAccessPolicy -AppId "<client-id-from-step-2>" `
    -PolicyScopeGroupId "VendorQuoteIngestScope" -AccessRight RestrictAccess `
    -Description "Limit Vendor Quote Ingest app to procurement mailbox only"
```

Verify with: `Test-ApplicationAccessPolicy -Identity "procurement@yourdomain.com" -AppId "<client-id>"` — should return `AccessAllowed`.

## 6. Add the `[graph]` section to `secrets.toml`

Edit `.streamlit/secrets.toml` (or whichever path `LOCAL_SECRETS_PATHS` resolves to first). Add:

```toml
[graph]
tenant_id = "00000000-0000-0000-0000-000000000000"
client_id = "00000000-0000-0000-0000-000000000000"
certificate_path = "C:/Users/alexh/vendor-quote-ingest.pem"
certificate_thumbprint = "ABCDEF0123456789ABCDEF0123456789ABCDEF01"
mailbox = "procurement@yourdomain.com"
```

Then verify:

```powershell
.\.venv\Scripts\python.exe -c "from secrets_loader import load_graph_settings; s = load_graph_settings(); print('ok' if s else 'missing keys')"
```
Expected: `ok`.

## 7. First manual run (backfill + verify)

```powershell
.\.venv\Scripts\python.exe vendor_quote_ingest.py --backfill-days 30 --verbose
```

Watch for:
- `seen=N matched=M rows=R` summary at the end.
- `data\vendor_quotes.json` created with at least one item key.
- `data\vendor_quote_cursor.json` written.

If you get `Graph auth failed`, recheck steps 2–4 (thumbprint, consent).

## 8. Register Task Scheduler entry

```powershell
$action = New-ScheduledTaskAction `
    -Execute "C:\Users\alexh\Downloads\mod\.venv\Scripts\python.exe" `
    -Argument "C:\Users\alexh\Downloads\mod\vendor_quote_ingest.py" `
    -WorkingDirectory "C:\Users\alexh\Downloads\mod"

$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(2) `
    -RepetitionInterval (New-TimeSpan -Minutes 15)

$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType S4U

Register-ScheduledTask `
    -TaskName "VendorQuoteIngest" `
    -Action $action `
    -Trigger $trigger `
    -Principal $principal `
    -Description "Pull vendor quotes from Outlook every 15 minutes"
```

Verify with `Get-ScheduledTaskInfo -TaskName "VendorQuoteIngest"`. Tail
`data\vendor_quote_ingest.log` after 20 minutes to confirm runs are happening.

## Certificate rotation (every 2 years)

The cert from step 1 expires 3 years out, but rotate at the 2-year mark to
avoid a surprise outage. To rotate:

1. Repeat step 1 with a new subject (`CN=vendor-quote-ingest-2028`) to generate
   a fresh cert.
2. Upload the new `.cer` via step 3 (Azure now holds two valid certs).
3. Update `certificate_path` and `certificate_thumbprint` in `secrets.toml`.
4. Run `vendor_quote_ingest.py --backfill-days 0 --verbose` to confirm the new
   cert authenticates.
5. In Azure, delete the old cert.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Graph auth failed: AADSTS70011` | Missing admin consent | Re-do step 4. |
| `Graph auth failed: AADSTS700027` | Cert thumbprint mismatch | Verify step 1's thumbprint matches step 3's upload. |
| `Graph auth failed: AADSTS700016` | App registration not yet propagated | Wait 5 min after step 2 and retry. |
| `Lock present at ...` | Prior run crashed mid-flight | Delete `data\vendor_quote_ingest.lock`. |
| Cursor never updates | Filter rejecting everything | Run with `--verbose`; inspect which messages were `seen` vs `matched`. |
| OpenAI extractor returns 0 rows | Prompt missing aliases | Add missing aliases to `data\vendor_quote_aliases.json`. |
| Refresh button shows "Already running" | Scheduled task and manual click overlapped | Wait for cron run to finish (≤2 min) then retry. |
