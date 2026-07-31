# Babel integration fork

This repository is a fork of
[NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent).
The upstream Git history is retained so authorship and provenance remain
traceable at the commit level.

Hermes Agent is distributed under the MIT License. The original copyright
notice and license terms remain in [`LICENSE`](LICENSE) and must accompany
source or substantial binary distributions of this fork.

## Babel-maintained changes

Middleware Technology maintains a small integration layer for Babel's desktop
runtime and Action Board execution model. The initial integration branch is
based on upstream commit `9fb40e6a3d6338b6a6a616010de7a16672148924` and adds:

- scoped tool and filesystem enforcement at the Hermes dispatch boundary;
- Action Board claim fencing and idempotent terminal transitions;
- host-managed process cleanup and restart behavior;
- configurable long-running agent iteration limits; and
- regression coverage for the integration contracts.

These changes do not alter the attribution or license of upstream code.
Babel-specific commits identify their accountable human contributor through
normal Git authorship. AI-assisted coding or review may be used as a tool, but
the human commit author remains responsible for the contribution; an AI system
is not listed as a copyright holder or co-author.

## Upstream policy

The `upstream` Git remote should track the original Nous Research repository.
Broadly useful fixes should be proposed upstream when practical. Babel-only
integration behavior may remain on a pinned fork commit so packaged releases
are reproducible while upstream work continues independently.
