# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kube CLI subcommand group with lazy-loaded subcommands."""

from __future__ import annotations

from cyclopts import App

app = App(name="kube", help="Kubernetes deployment and management commands")

app.command("aiperf.cli_commands.kube.attach:app", name="attach")
app.command("aiperf.cli_commands.kube.cancel:app", name="cancel")
app.command("aiperf.cli_commands.kube.debug:app", name="debug")
app.command("aiperf.cli_commands.kube.cleanup:app", name="cleanup")
app.command("aiperf.cli_commands.kube.shutdown:app", name="shutdown")
app.command("aiperf.cli_commands.kube.delete:app", name="delete")
app.command("aiperf.cli_commands.kube.generate:app", name="generate")
app.command("aiperf.cli_commands.kube.init:app", name="init")
app.command("aiperf.cli_commands.kube.logs:app", name="logs")
app.command("aiperf.cli_commands.kube.preflight:app", name="preflight")
app.command("aiperf.cli_commands.kube.profile:app", name="profile")
app.command("aiperf.cli_commands.kube.results:app", name="results")
app.command("aiperf.cli_commands.kube.list_:app", name="list")
app.command("aiperf.cli_commands.kube.validate:app", name="validate")
app.command("aiperf.cli_commands.kube.watch:app", name="watch")
