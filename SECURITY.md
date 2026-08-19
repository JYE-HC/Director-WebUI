# Security policy

Director is a ComfyUI plugin (distributed as DirectorDeck) intended for one
trusted user on a local machine or a trusted private network. It has no
built-in login, authorization boundary or TLS termination. The embedded
backend listens only on a loopback internal port; user-facing traffic reaches
it same-origin through ComfyUI under `/directordeck/`. Keep ComfyUI itself on
loopback; before remote access, place ComfyUI behind an authenticated TLS
reverse proxy with source restrictions. Do not expose ComfyUI or Director
directly to the public Internet.

After the GitHub repository is created, report vulnerabilities privately with
GitHub's **Security → Report a vulnerability** flow. Do not include prompts,
media, database contents, access tokens, private addresses, GPU UUIDs or full
diagnostic logs in a public issue. Until private reporting is configured, do
not publish exploit details; contact the repository owner through their chosen
private channel.
