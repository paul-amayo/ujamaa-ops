# ujamaa.ai migration — prepared groundwork (companion to plans/web_deployment.md)

State captured 2026-09-22: all four servers UP (8001 viewer/pillars, 8002 query,
8003 Adinkra panel, 8004 GPU render ×13 workers + ollama 0.32.15). They are shell-
children today — a reboot or closed shell kills them; that is what `units/` fixes.

## Box-side steps (safe order; ~1 h of work, no GPU impact)
1. `sudo cp units/*.service /etc/systemd/system/ && sudo systemctl daemon-reload`
2. Stop the shell-started servers, then `sudo systemctl enable --now ujamaa-{viewer,query,adinkra,render}`
   — units bind **127.0.0.1** (the 0.0.0.0-on-public-IP finding). CAVEAT: if the V100
   VM calls this box's services over the LAN, keep 0.0.0.0 for those until the call
   moves into the tunnel; SSH tunnels (`ssh -L`) are unaffected by loopback binding.
3. `curl -fsSL https://pkg.cloudflare.com/... cloudflared` install → `cloudflared tunnel login`
   (opens a browser auth — PAUL does this step) → create tunnel + DNS routes →
   drop `cloudflared-config.yml.template` into /etc/cloudflared/config.yml →
   `sudo systemctl enable --now cloudflared`.
4. Cloudflare Access: one app per hostname, email-OTP allow-list.

## Paul-side decisions (blockers, in order)
- [ ] Move ujamaa.ai DNS to a free Cloudflare zone (GoDaddy stays registrar): add site
      in Cloudflare, swap the two nameservers in GoDaddy. ~30 min + propagation.
- [ ] Access allow-list: which emails (us + WWW + collaborators)?
- [ ] Evening fleet policy vs ollama: exempt the demo ollama, or publish demo hours?
- [ ] P2 measurement to schedule: client-side `.splat` rendering at demo scale
      (option A — would take the GPU out of the demo path entirely except ollama).
- [ ] Does the V100 VM consume any of 8001–8004 over the LAN? (Determines whether
      loopback rebind can be immediate.)

Nothing here is installed or enabled yet — files only; installation needs sudo and
Paul's Cloudflare/GoDaddy logins, per the deployment roadmap's P0.
