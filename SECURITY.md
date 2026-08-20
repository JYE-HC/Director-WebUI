# Security policy

Director is a ComfyUI plugin (distributed as DirectorDeck) intended for one
trusted user on a local machine or a trusted private network. It has no
built-in login, authorization boundary or TLS termination. The embedded
backend listens only on a loopback internal port; user-facing traffic reaches
it same-origin through ComfyUI under `/directordeck/`. Keep ComfyUI itself on
loopback; before remote access, place ComfyUI behind an authenticated TLS
reverse proxy with source restrictions. Do not expose ComfyUI or Director
directly to the public Internet.

Report vulnerabilities privately with the repository's GitHub **Security →
Report a vulnerability** flow. If that private form is unavailable, do not
open a public issue with exploit details; contact the repository owner through
their GitHub profile first. Never include prompts, media, database contents,
access tokens, private addresses, GPU UUIDs or full diagnostic logs in a public
issue.
