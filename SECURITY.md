# Security policy

Director Web 0.1.x is intended for one trusted user on a local machine or a
trusted private network. It has no built-in login, authorization boundary or
TLS termination. Keep the default loopback listeners; before remote access,
place both services behind an authenticated TLS reverse proxy with source
restrictions. Do not expose ComfyUI or Director directly to the public Internet.

The installer never pulls or patches ComfyUI, never downloads models, never
deletes a conflicting custom node, and never stops ComfyUI. Explicit node
replacement keeps the previous directory under `.director-backups/`.

After the GitHub repository is created, report vulnerabilities privately with
GitHub's **Security → Report a vulnerability** flow. Do not include prompts,
media, database contents, access tokens, private addresses, GPU UUIDs or full
diagnostic logs in a public issue. Until private reporting is configured, do
not publish exploit details; contact the repository owner through their chosen
private channel.
