import type { ChatClient, ChatMessage } from "@chatman-media/rag";
import { describe, expect, it } from "bun:test";
import {
  CoachAnalyzer,
  type CoachAnalyzerLead,
  type CoachAnalyzerMessage,
  extractUserAssistantPairs,
} from "../coach-analyzer.ts";

function msg(
  opts: { id: number; role: CoachAnalyzerMessage["role"]; text: string },
): CoachAnalyzerMessage {
  return { id: opts.id, role: opts.role, text: opts.text };
}

describe("extractUserAssistantPairs", () => {
  it("Собирает alternating user→assistant пары", () => {
    const out = extractUserAssistantPairs([
      msg({ id: 1, role: "user", text: "привет" }),
      msg({ id: 2, role: "assistant", text: "здравствуй" }),
      msg({ id: 3, role: "user", text: "сколько" }),
      msg({ id: 4, role: "assistant", text: "уточню" }),
    ]);
    expect(out).toEqual([
      { userText: "привет", assistantText: "здравствуй", assistantMessageId: 2 },
      { userText: "сколько", assistantText: "уточню", assistantMessageId: 4 },
    ]);
  });

  it("human-роль считается assistant'ом (operator reply)", () => {
    const out = extractUserAssistantPairs([
      msg({ id: 1, role: "user", text: "?" }),
      msg({ id: 2, role: "human", text: "manual reply" }),
    ]);
    expect(out).toEqual([{ userText: "?", assistantText: "manual reply", assistantMessageId: 2 }]);
  });

  it("два user подряд (бот не ответил) — пропускает первый", () => {
    const out = extractUserAssistantPairs([
      msg({ id: 1, role: "user", text: "first" }),
      msg({ id: 2, role: "user", text: "second" }),
      msg({ id: 3, role: "assistant", text: "reply" }),
    ]);
    expect(out).toEqual([{ userText: "second", assistantText: "reply", assistantMessageId: 3 }]);
  });

  it("system-роль игнорируется", () => {
    const out = extractUserAssistantPairs([
      msg({ id: 1, role: "system", text: "init" }),
      msg({ id: 2, role: "user", text: "hi" }),
      msg({ id: 3, role: "assistant", text: "hi back" }),
    ]);
    expect(out).toEqual([{ userText: "hi", assistantText: "hi back", assistantMessageId: 3 }]);
  });

  it("пустой text → skip pair", () => {
    const out = extractUserAssistantPairs([
      msg({ id: 1, role: "user", text: "" }),
      msg({ id: 2, role: "assistant", text: "reply" }),
    ]);
    expect(out).toEqual([]);
  });
});

class FakeChat implements ChatClient {
  public calls = 0;
  public lastMessages: ChatMessage[] | undefined;
  constructor(private readonly outputs: string[]) {}
  async complete(messages: ChatMessage[]): Promise<string> {
    this.lastMessages = messages;
    const out = this.outputs[this.calls] ?? "[]";
    this.calls += 1;
    return out;
  }
}

class FakeSkillOutcomes {
  public recorded: Array<{
    leadId: number;
    skillSlug: string;
    outcome: string;
    messageId: number;
  }> = [];
  private seen = new Set<string>();
  async record(opts: {
    leadId: number;
    skillSlug: string;
    outcome: "won" | "lost" | "draw";
    source: string;
    conversationId: number;
    messageId: number;
    styleSlug: string | null;
    nowEpoch: number;
  }): Promise<boolean> {
    const key = `${opts.leadId}:${opts.skillSlug}:${opts.source}`;
    if (this.seen.has(key)) return false;
    this.seen.add(key);
    this.recorded.push({
      leadId: opts.leadId,
      skillSlug: opts.skillSlug,
      outcome: opts.outcome,
      messageId: opts.messageId,
    });
    return true;
  }
}

class FakeMessages {
  constructor(private readonly rows: readonly CoachAnalyzerMessage[]) {}
  async recent(_conversationId: number, _limit: number): Promise<readonly CoachAnalyzerMessage[]> {
    return [...this.rows];
  }
}

describe("CoachAnalyzer", () => {
  const lead: CoachAnalyzerLead = { id: 42, tenantId: 1 };

  it("анализирует пары и пишет skill_outcomes", async () => {
    const chat = new FakeChat(['["social-proof-stat","tactical-empathy"]', '["mirroring"]']);
    const analyzer = new CoachAnalyzer({
      availableSlugs: ["social-proof-stat", "tactical-empathy", "mirroring"],
      resolveChat: () => chat,
    });
    const messages = new FakeMessages([
      msg({ id: 1, role: "user", text: "сколько платят" }),
      msg({ id: 2, role: "assistant", text: "70% наших закрывают за месяц" }),
      msg({ id: 3, role: "user", text: "не уверена" }),
      msg({ id: 4, role: "assistant", text: "не уверена?" }),
    ]);
    const outcomes = new FakeSkillOutcomes();
    const result = await analyzer.analyzeLead(
      { messages, skillOutcomes: outcomes },
      {
        lead,
        conversationId: 100,
        styleSlug: "alina-infinity-v1",
        outcome: "won",
        source: "lead_submitted",
        nowEpoch: 1700000000,
      },
    );
    expect(result.pairsAnalyzed).toBe(2);
    expect(result.outcomesRecorded).toBe(3);
    expect(result.outcomesDuplicate).toBe(0);
    expect(chat.calls).toBe(2);
    expect(outcomes.recorded.map((r) => r.skillSlug).sort()).toEqual([
      "mirroring",
      "social-proof-stat",
      "tactical-empathy",
    ]);
  });

  it("idempotency: повторный analyze того же lead → 0 newly recorded", async () => {
    const chat = new FakeChat(['["mirroring"]', '["mirroring"]']);
    const analyzer = new CoachAnalyzer({
      availableSlugs: ["mirroring"],
      resolveChat: () => chat,
    });
    const messages = new FakeMessages([
      msg({ id: 1, role: "user", text: "x" }),
      msg({ id: 2, role: "assistant", text: "x?" }),
    ]);
    const outcomes = new FakeSkillOutcomes();
    await analyzer.analyzeLead(
      { messages, skillOutcomes: outcomes },
      {
        lead,
        conversationId: 100,
        styleSlug: null,
        outcome: "lost",
        source: "lead_rejected",
        nowEpoch: 0,
      },
    );
    const r2 = await analyzer.analyzeLead(
      { messages, skillOutcomes: outcomes },
      {
        lead,
        conversationId: 100,
        styleSlug: null,
        outcome: "lost",
        source: "lead_rejected",
        nowEpoch: 0,
      },
    );
    expect(r2.outcomesRecorded).toBe(0);
    expect(r2.outcomesDuplicate).toBeGreaterThan(0);
  });

  it("нет user→assistant пар → 0 анализов, 0 LLM call'ов", async () => {
    const chat = new FakeChat([]);
    const analyzer = new CoachAnalyzer({
      availableSlugs: ["mirroring"],
      resolveChat: () => chat,
    });
    const messages = new FakeMessages([
      msg({ id: 1, role: "system", text: "init" }),
      msg({ id: 2, role: "user", text: "lone user" }),
    ]);
    const outcomes = new FakeSkillOutcomes();
    const result = await analyzer.analyzeLead(
      { messages, skillOutcomes: outcomes },
      {
        lead,
        conversationId: 100,
        styleSlug: null,
        outcome: "draw",
        source: "lead_ghosted",
        nowEpoch: 0,
      },
    );
    expect(result.pairsAnalyzed).toBe(0);
    expect(chat.calls).toBe(0);
  });
});
