import express from "express";
import multer from "multer";
import { createRequire } from "node:module";
import { loadEnvFile } from "node:process";

const require = createRequire(import.meta.url);
const { PDFParse } = require("pdf-parse");

try {
  loadEnvFile();
} catch {
  // .env is optional; environment variables can still be exported by the shell.
}

const app = express();
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 12 * 1024 * 1024, files: 8 }
});

const PORT = process.env.PORT || 8787;
const GROQ_API_KEY = process.env.GROQ_API_KEY || "";
const XAI_API_KEY = process.env.XAI_API_KEY || "";
const PROVIDER = process.env.LLM_PROVIDER || (GROQ_API_KEY || XAI_API_KEY.startsWith("gsk_") ? "groq" : "xai");
const ACTIVE_API_KEY = PROVIDER === "groq" ? (GROQ_API_KEY || XAI_API_KEY) : XAI_API_KEY;
const ACTIVE_MODEL =
  process.env.LLM_MODEL ||
  (PROVIDER === "groq"
    ? process.env.GROQ_MODEL || "openai/gpt-oss-120b"
    : process.env.XAI_MODEL || "grok-4.7");
const CHAT_COMPLETIONS_URL =
  PROVIDER === "groq"
    ? "https://api.groq.com/openai/v1/chat/completions"
    : "https://api.x.ai/v1/chat/completions";

app.use(express.json({ limit: "18mb" }));

const financeTerms = [
  "finance", "financial", "money", "income", "salary", "tax", "itr", "return",
  "filing", "deduction", "investment", "loan", "emi", "credit", "debit",
  "insurance", "budget", "expense", "savings", "bank", "mutual", "fund",
  "stock", "equity", "debt", "capital", "gain", "loss", "gst", "tds",
  "form 16", "ais", "26as", "pan", "hra", "80c", "80d", "nps", "pf",
  "retirement", "pension", "invoice", "accounting", "asset", "liability",
  "net worth", "interest", "inflation", "sip", "portfolio", "balance sheet",
  "ebitda", "ebit", "profit", "revenue", "turnover", "margin", "fy", "fiscal",
  "quarter", "annual", "crore", "lakh", "rs", "₹", "inr", "eps", "cash flow",
  "p&l", "income statement", "annual report", "earnings", "valuation", "capex",
  "opex", "dividend", "shareholder", "market cap", "working capital"
];

const documents = new Map();

function getSession(sessionId) {
  const key = sessionId || "demo";
  if (!documents.has(key)) documents.set(key, []);
  return documents.get(key);
}

function isFinanceQuestion(text = "") {
  const lower = text.toLowerCase();
  return (
    financeTerms.some((term) => lower.includes(term)) ||
    /fy\s?\d{2,4}|q[1-4]\s?fy|₹|rs\.?\s?\d|inr\s?\d/i.test(text)
  );
}

function chunkText(text, source) {
  const clean = text
    .replace(/\r/g, "\n")
    .replace(/[ \t]+/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
  const chunks = [];
  for (let i = 0; i < clean.length; i += 1400) {
    const body = clean.slice(i, i + 1700);
    if (body.length > 20) chunks.push({ source, body });
  }
  return chunks;
}

function uniqueSources(context) {
  return [...new Set(context.map((item) => item.source))];
}

function getRelevantSnippets(context, limit = 3) {
  return context
    .slice(0, limit)
    .map((item) => {
      const compact = item.body.replace(/\s+/g, " ").trim();
      return `- ${compact.slice(0, 420)}${compact.length > 420 ? "..." : ""}`;
    })
    .join("\n");
}

function getQuestionTerms(question) {
  const stopWords = new Set([
    "what", "was", "were", "the", "and", "for", "from", "with", "that", "this",
    "how", "why", "who", "when", "where", "which", "into", "about", "please",
    "tell", "explain", "show", "give", "year", "crore"
  ]);
  return [...new Set(question.toLowerCase().match(/[a-z0-9]{3,}/g) || [])]
    .filter((word) => !stopWords.has(word));
}

const metricAliases = [
  { key: "ebitda", names: ["ebitda"] },
  { key: "revenue", names: ["revenue", "turnover", "sales", "income from operations"] },
  { key: "pat", names: ["pat", "profit after tax", "net profit"] },
  { key: "pbt", names: ["pbt", "profit before tax"] },
  { key: "ebit", names: ["ebit"] },
  { key: "eps", names: ["eps", "earnings per share"] },
  { key: "margin", names: ["margin"] },
  { key: "capex", names: ["capex", "capital expenditure"] },
  { key: "debt", names: ["debt", "borrowings"] },
  { key: "cash flow", names: ["cash flow", "operating cash flow", "free cash flow"] }
];

function getRequestedMetric(question) {
  const lower = question.toLowerCase();
  if (lower.includes("margin")) {
    return metricAliases.find((metric) => metric.key === "margin");
  }
  return metricAliases.find((metric) => metric.names.some((name) => lower.includes(name)));
}

function getRequestedYear(question) {
  const fyMatch = question.match(/fy\s*(20)?(\d{2,4})/i);
  if (fyMatch) return fyMatch[2].length === 2 ? `20${fyMatch[2]}` : fyMatch[2];
  const yearMatch = question.match(/\b20\d{2}\b/);
  return yearMatch?.[0] || null;
}

function parseAmountFromLine(line, metricIndex = 0) {
  const patterns = [
    /(?<currency>₹|rs\.?|inr)?\s*(?<value>[\d,]+(?:\.\d+)?)\s*(?<unit>crore|cr|lakh|million|mn|billion|bn)?\b/gi,
    /(?<value>[\d,]+(?:\.\d+)?)\s*(?<unit>%)/gi
  ];
  const candidates = [];

  for (const pattern of patterns) {
    for (const match of line.matchAll(pattern)) {
      const value = match.groups?.value;
      if (!value || /^\d{4}$/.test(value.replace(/,/g, ""))) continue;
      const unit = normalizeUnit(match.groups?.unit || "");
      const currency = normalizeCurrency(match.groups?.currency || "");
      const distance = Math.abs((match.index || 0) - metricIndex);
      candidates.push({ value, unit, currency, distance });
    }
  }

  candidates.sort((a, b) => a.distance - b.distance);
  return candidates[0] || null;
}

function normalizeCurrency(currency) {
  if (!currency) return "";
  return currency.toLowerCase().includes("inr") || currency.toLowerCase().includes("rs") || currency === "₹"
    ? "₹"
    : currency;
}

function normalizeUnit(unit) {
  const lower = unit.toLowerCase();
  if (lower === "cr") return "crore";
  if (lower === "mn") return "million";
  if (lower === "bn") return "billion";
  return lower;
}

function inferAmountUnit(text) {
  const lower = text.toLowerCase();
  const hints = [
    { unit: "million", pattern: /(₹|rs\.?|inr)?\s*(in)?\s*(million|mn)|amounts?\s+(are\s+)?(in\s+)?(₹|rs\.?|inr)?\s*(million|mn)/i },
    { unit: "crore", pattern: /(₹|rs\.?|inr)?\s*(in)?\s*(crore|cr)|amounts?\s+(are\s+)?(in\s+)?(₹|rs\.?|inr)?\s*(crore|cr)/i },
    { unit: "lakh", pattern: /(₹|rs\.?|inr)?\s*(in)?\s*lakh|amounts?\s+(are\s+)?(in\s+)?(₹|rs\.?|inr)?\s*lakh/i },
    { unit: "billion", pattern: /(₹|rs\.?|inr)?\s*(in)?\s*(billion|bn)|amounts?\s+(are\s+)?(in\s+)?(₹|rs\.?|inr)?\s*(billion|bn)/i }
  ];
  const found = hints.find((hint) => hint.pattern.test(lower));
  return found?.unit || "";
}

function formatAmount(amount, inferredUnit, question) {
  const unit = amount.unit || inferredUnit;
  const currency = amount.currency || (question.includes("₹") || /\b(rs|inr|crore|lakh)\b/i.test(question) ? "₹" : "");
  const value = amount.value;
  const numericValue = Number(value.replace(/,/g, ""));

  if (unit === "%") {
    return { text: `${value}%`, caveat: "" };
  }

  if (!unit) {
    return {
      text: `${currency}${value}`,
      caveat: "The extracted line did not include a unit, so please verify whether the PDF table is in million, crore, lakh, or another unit."
    };
  }

  const text = `${currency}${value} ${unit}`.trim();
  if (unit === "million" && Number.isFinite(numericValue)) {
    return {
      text: `${text} (about ₹${(numericValue / 10).toLocaleString("en-IN", { maximumFractionDigits: 2 })} crore)`,
      caveat: "Converted using 1 crore = 10 million."
    };
  }

  if (unit === "lakh" && Number.isFinite(numericValue)) {
    return {
      text: `${text} (about ₹${(numericValue / 100).toLocaleString("en-IN", { maximumFractionDigits: 2 })} crore)`,
      caveat: "Converted using 1 crore = 100 lakh."
    };
  }

  return { text, caveat: "" };
}

function getEvidenceLines(question, context, limit = 6) {
  const terms = getQuestionTerms(question);
  const lines = [];

  for (const item of context) {
    const parts = item.body
      .split(/\n|(?<=\.)\s+|(?<=;)\s+/)
      .map((line) => line.replace(/\s+/g, " ").trim())
      .filter((line) => line.length > 12);

    for (const line of parts) {
      const lower = line.toLowerCase();
      let score = 0;
      for (const term of terms) {
        if (lower.includes(term)) score += term.length > 4 ? 2 : 1;
      }
      if (/[₹$]|\b(rs|inr|crore|cr|lakh|million|billion|%)\b/i.test(line)) score += 1;
      if (score > 0) lines.push({ source: item.source, line, score });
    }
  }

  return lines
    .sort((a, b) => b.score - a.score)
    .slice(0, limit);
}

function formatEvidence(lines) {
  return lines
    .map((item) => `- ${item.line.slice(0, 520)}${item.line.length > 520 ? "..." : ""}`)
    .join("\n");
}

function buildDocumentAnswer(question, context) {
  const evidence = getEvidenceLines(question, context, 6);
  if (!evidence.length) {
    return {
      answer: `I found uploaded document context, but I could not find lines that clearly match your question.\n\nTry asking with the exact term from the PDF, such as EBITDA, revenue, PAT, EPS, tax, deduction, or FY2025.`,
      sources: uniqueSources(context)
    };
  }

  return {
    answer: [
      "I found these matching lines in your uploaded document:",
      "",
      formatEvidence(evidence),
      "",
      "Local mode is extractive, so I will not invent a number if the PDF text is unclear. For stronger table reading and fuller explanations, add `XAI_API_KEY` so Grok can reason over the retrieved context."
    ].join("\n"),
    sources: uniqueSources(context)
  };
}

function extractMetricAnswer(question, context) {
  const requestedMetric = getRequestedMetric(question);
  if (!requestedMetric) return null;

  const requestedYear = getRequestedYear(question);
  const yearNeedles = requestedYear
    ? [`fy${requestedYear}`, `fy ${requestedYear}`, requestedYear]
    : [];
  const sourceText = context.map((item) => item.body).join("\n");
  const yearsInContext = [...new Set(sourceText.match(/20\d{2}/g) || [])];
  const canUseDocumentYear =
    requestedYear &&
    yearsInContext.includes(requestedYear) &&
    yearsInContext.length === 1;
  const lines = sourceText
    .split(/\n|(?<=\.)\s+/)
    .map((line) => line.replace(/\s+/g, " ").trim())
    .filter(Boolean);

  const candidatesLines = [];
  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    const lowerLine = line.toLowerCase().replace(/\s+/g, "");
    const metricName = requestedMetric.names.find((name) => lowerLine.includes(name.replace(/\s+/g, "")));
    if (!metricName) continue;

    const nearbyText = lines.slice(Math.max(0, index - 1), index + 2).join(" | ");
    const lower = nearbyText.toLowerCase().replace(/\s+/g, "");
    const hasMetric = requestedMetric.names.some((name) => lower.includes(name.replace(/\s+/g, "")));
    const hasYear =
      !yearNeedles.length ||
      yearNeedles.some((year) => lower.includes(year.replace(/\s+/g, ""))) ||
      canUseDocumentYear;
    if (hasMetric && hasYear) candidatesLines.push({ line, nearbyText, metricName });
  }

  for (const candidateLine of candidatesLines) {
    const windowText = candidateLine.line;
    const lower = windowText.toLowerCase();
    const metricIndex = lower.indexOf(candidateLine.metricName);
    const amount = parseAmountFromLine(windowText, metricIndex);
    if (amount) {
      const inferredUnit = inferAmountUnit(`${candidateLine.nearbyText}\n${sourceText.slice(0, 1200)}`);
      const formatted = formatAmount(amount, inferredUnit, question);
      return {
        answer: [
          `I found ${requestedYear ? `FY${requestedYear} ` : ""}${requestedMetric.key.toUpperCase()} as ${formatted.text} in the uploaded document.`,
          formatted.caveat ? `Unit note: ${formatted.caveat}` : "",
          "",
          `Source snippet: ${candidateLine.nearbyText}`
        ].filter(Boolean).join("\n"),
        sources: uniqueSources(context)
      };
    }
  }

  return {
    answer: [
      `I found ${requestedMetric.key.toUpperCase()}-related context, but I cannot confidently identify the exact ${requestedYear ? `FY${requestedYear} ` : ""}${requestedMetric.key.toUpperCase()} value from the extracted PDF text.`,
      "",
      "Closest matching lines:",
      formatEvidence(getEvidenceLines(question, context, 5)),
      "",
      "This usually happens when the value is inside a PDF table whose text extraction order is messy. Connect `XAI_API_KEY` for stronger table reasoning."
    ].join("\n"),
    sources: uniqueSources(context)
  };
}

function scoreChunk(question, chunk) {
  const words = new Set(getQuestionTerms(question));
  const body = chunk.body.toLowerCase();
  const lowerQuestion = question.toLowerCase();
  let score = 0;
  for (const word of words) {
    if (body.includes(word)) score += 1;
  }
  for (const metric of metricAliases) {
    if (metric.names.some((name) => lowerQuestion.includes(name))) {
      for (const name of metric.names) {
        if (body.includes(name)) score += 4;
      }
    }
  }
  const year = getRequestedYear(question);
  if (year && body.includes(year)) score += 3;
  if (/[₹$]|\b(rs|inr|crore|cr|lakh|million|billion|%)\b/i.test(chunk.body)) score += 1;
  return score;
}

function retrieveContext(sessionId, question) {
  return getSession(sessionId)
    .map((chunk) => ({ ...chunk, score: scoreChunk(question, chunk) }))
    .filter((chunk) => chunk.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, 10);
}

async function extractFile(file) {
  if (file.mimetype === "application/pdf" || file.originalname.toLowerCase().endsWith(".pdf")) {
    const parser = new PDFParse({ data: file.buffer });
    try {
      const parsed = await parser.getText();
      let tableText = "";
      try {
        const tableResult = await parser.getTable();
        tableText = (tableResult.pages || [])
          .flatMap((page, pageIndex) =>
            (page.tables || []).flatMap((table, tableIndex) => [
              `Table from page ${pageIndex + 1}, table ${tableIndex + 1}:`,
              ...table.map((row) => row.map((cell) => String(cell ?? "").trim()).join(" | "))
            ])
          )
          .filter(Boolean)
          .join("\n");
      } catch {
        tableText = "";
      }

      return {
        type: "text",
        source: file.originalname,
        text: [parsed.text || "", tableText ? `\n\nExtracted tables:\n${tableText}` : ""].join("")
      };
    } finally {
      await parser.destroy();
    }
  }

  if (file.mimetype.startsWith("text/")) {
    return {
      type: "text",
      source: file.originalname,
      text: file.buffer.toString("utf8")
    };
  }

  if (file.mimetype.startsWith("image/")) {
    return {
      type: "image",
      source: file.originalname,
      mimeType: file.mimetype,
      dataUrl: `data:${file.mimetype};base64,${file.buffer.toString("base64")}`
    };
  }

  return {
    type: "unsupported",
    source: file.originalname,
    text: ""
  };
}

function fallbackAnswer(question, context) {
  const contextText = context.map((item) => item.body).join(" ");
  const lower = `${question} ${contextText}`.toLowerCase();

  if (!isFinanceQuestion(question)) {
    return {
      answer: "I can help only with finance, tax, banking, investing, budgeting, insurance, loans, and ITR-related questions. Please ask something in that area.",
      sources: []
    };
  }

  if (lower.includes("itr") || lower.includes("income tax return") || lower.includes("filing")) {
    return {
      answer: "ITR filing is the yearly process of telling the Income Tax Department your income, deductions, taxes already paid, and refund or tax payable. For most salaried people, the simple flow is: collect Form 16, check AIS and Form 26AS, choose old or new regime, add income from salary, interest, capital gains or house property, claim eligible deductions if using the old regime, verify TDS, pay any remaining tax, file the matching ITR form, then e-verify it. This prototype can use uploaded PDFs or notes as context, but please verify final filing details with a qualified tax professional or the official income tax portal.",
      sources: context.map((item) => item.source)
    };
  }

  if (context.length) {
    const metricAnswer = extractMetricAnswer(question, context);
    if (metricAnswer) {
      return metricAnswer;
    }

    return buildDocumentAnswer(question, context);
  }

  return {
    answer: "This is a finance question, but I do not have matching uploaded document context available right now. Please add the PDF again with Add to RAG, then ask the EBITDA question once the upload message confirms it was added.",
    sources: []
  };
}

async function askLlm({ question, context, images }) {
  const contextBlock = context.length
    ? context.map((item, index) => `[${index + 1}] ${item.source}\n${item.body}`).join("\n\n")
    : "No uploaded text context matched this question.";

  const content = [
    {
      type: "text",
      text: `Question: ${question}\n\nRetrieved user context:\n${contextBlock}`
    },
    ...images.slice(0, 3).map((image) => ({
      type: "image_url",
      image_url: { url: image.dataUrl }
    }))
  ];

  const response = await fetch(CHAT_COMPLETIONS_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${ACTIVE_API_KEY}`
    },
    body: JSON.stringify({
      model: ACTIVE_MODEL,
      temperature: 0.2,
      max_tokens: 1600,
      messages: [
        {
          role: "system",
          content: [
            "You are FinWise, a finance-only RAG assistant for common people.",
            "Answer only questions about personal finance, taxes, ITR filing, banking, loans, insurance, budgeting, investing, retirement, accounting, or uploaded financial documents.",
            "If the user asks anything outside finance, politely refuse in one sentence and invite a finance-related question.",
            "Give complete but concise answers in plain language.",
            "For document questions, answer from retrieved context first, cite the source filename, and quote the short supporting line or table row when useful.",
            "For finance metrics, be careful not to confuse revenue, EBITDA, PAT, EPS, margin, or cash flow. If the exact value is unclear, say so.",
            "Use this structure when helpful: Direct answer, Why it matters, Evidence, Caveat.",
            "Avoid legal certainty. Encourage checking official portals or qualified professionals for final tax, legal, or investment decisions.",
            "When context is insufficient, say what is missing instead of inventing facts."
          ].join(" ")
        },
        { role: "user", content }
      ]
    })
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(formatProviderError(response.status, errorText));
  }

  const data = await response.json();
  return data.choices?.[0]?.message?.content || "I could not generate a response.";
}

function formatProviderError(status, errorText) {
  let message = errorText;
  try {
    const parsed = JSON.parse(errorText);
    message = parsed.error?.message || parsed.message || errorText;
  } catch {
    message = errorText;
  }

  if (status === 401 || status === 403) {
    return `${PROVIDER === "groq" ? "Groq" : "xAI"} rejected the API key. Please check that the key is valid for ${PROVIDER === "groq" ? "console.groq.com" : "console.x.ai"}.`;
  }
  if (status === 404 || /model/i.test(message)) {
    return `${PROVIDER === "groq" ? "Groq" : "xAI"} model error for "${ACTIVE_MODEL}". Details: ${message}`;
  }
  if (status === 429) {
    return `${PROVIDER === "groq" ? "Groq" : "xAI"} rate limit or quota reached. Please wait or check billing/limits.`;
  }
  return `${PROVIDER === "groq" ? "Groq" : "xAI"} API error ${status}: ${message}`;
}

app.get("/api/health", (_req, res) => {
  res.json({
    ok: true,
    llmConfigured: Boolean(ACTIVE_API_KEY),
    grokConfigured: Boolean(ACTIVE_API_KEY),
    provider: PROVIDER,
    model: ACTIVE_MODEL
  });
});

app.post("/api/upload", upload.array("files", 8), async (req, res) => {
  try {
    const sessionId = req.body.sessionId || "demo";
    const userDocs = getSession(sessionId);
    const added = [];

    for (const file of req.files || []) {
      const extracted = await extractFile(file);
      if (extracted.type === "text" && extracted.text.trim()) {
        const chunks = chunkText(extracted.text, extracted.source);
        userDocs.push(...chunks);
        added.push({ name: extracted.source, type: "text", chunks: chunks.length });
      } else if (extracted.type === "image") {
        userDocs.push({ source: extracted.source, body: `Image uploaded: ${extracted.source}`, image: extracted });
        added.push({ name: extracted.source, type: "image", chunks: 1 });
      } else {
        added.push({ name: extracted.source, type: "unsupported", chunks: 0 });
      }
    }

    if (req.body.text?.trim()) {
      const chunks = chunkText(req.body.text, "Pasted note");
      userDocs.push(...chunks);
      added.push({ name: "Pasted note", type: "text", chunks: chunks.length });
    }

    res.json({ ok: true, added, totalChunks: userDocs.length });
  } catch (error) {
    res.status(500).json({ ok: false, error: error.message });
  }
});

app.post("/api/chat", async (req, res) => {
  try {
    const { sessionId = "demo", question = "" } = req.body;
    const trimmedQuestion = question.trim();
    if (!trimmedQuestion) {
      return res.status(400).json({ ok: false, error: "Question is required." });
    }

    const imageDocs = getSession(sessionId)
      .filter((item) => item.image)
      .map((item) => item.image);
    const context = retrieveContext(sessionId, trimmedQuestion);

    if (!isFinanceQuestion(trimmedQuestion)) {
      return res.json(fallbackAnswer(trimmedQuestion, []));
    }

    if (!ACTIVE_API_KEY) {
      return res.json({ ...fallbackAnswer(trimmedQuestion, context), mode: "local-fallback" });
    }

    const answer = await askLlm({ question: trimmedQuestion, context, images: imageDocs });
    res.json({
      answer,
      sources: [...new Set(context.map((item) => item.source))],
      mode: "grok"
    });
  } catch (error) {
    res.status(500).json({ ok: false, error: error.message });
  }
});

app.listen(PORT, () => {
  console.log(`FinWise API running on http://127.0.0.1:${PORT}`);
});
