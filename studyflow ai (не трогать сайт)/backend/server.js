// ============================================================
// StudyFlow AI — серверная часть (референс, без сборки)
//
// Что делает:
//  - держит ключ AI-модели ТОЛЬКО на сервере (переменные окружения);
//  - /api/chat, /api/quiz, /api/plan, /api/cards — генерация через AI;
//  - анонимные сессии (токен, без личных данных);
//  - тарифы Free/Premium и дневные лимиты проверяются НА СЕРВЕРЕ;
//  - активация Premium по промокоду (готова к замене на платёжный вебхук);
//  - rate limiting, валидация входа, единая обработка ошибок.
//
// Запуск: npm install && npm start   (см. README.md)
// ============================================================

import express from 'express';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ---------- конфигурация ----------
const PORT = Number(process.env.PORT || 3000);
const PREMIUM_CODE = process.env.PREMIUM_CODE || '';
const ALLOW_CORS = process.env.ALLOW_CORS === '1';
const RATE_PER_MIN = Number(process.env.RATE_LIMIT_PER_MIN || 90);
const AI_RATE_PER_MIN = Number(process.env.AI_RATE_LIMIT_PER_MIN || 15);
const AI_PROVIDER = (process.env.AI_PROVIDER || (process.env.OPENAI_API_KEY ? 'openai' : 'zai')).toLowerCase();
const OPENAI_KEY = process.env.OPENAI_API_KEY || '';
const OPENAI_BASE = (process.env.OPENAI_BASE_URL || 'https://api.openai.com/v1').replace(/\/+$/, '');
const OPENAI_MODEL = process.env.OPENAI_MODEL || 'gpt-4o-mini';

const LIMITS = {
  free: {
    chat: Number(process.env.CHAT_LIMIT_FREE ?? 10),
    quiz: Number(process.env.QUIZ_LIMIT_FREE ?? 5),
    plan: 0,
    cards: 0,
  },
  premium: {
    chat: Number(process.env.CHAT_LIMIT_PREMIUM ?? 100),
    quiz: Number(process.env.QUIZ_LIMIT_PREMIUM ?? 50),
    plan: Number(process.env.PLAN_LIMIT_PREMIUM ?? 20),
    cards: Number(process.env.CARDS_LIMIT_PREMIUM ?? 30),
  },
};

const SESSIONS_FILE = path.join(__dirname, 'sessions.json');

// ---------- AI-провайдеры ----------
let zaiPromise = null;
async function getZai() {
  if (!zaiPromise) {
    zaiPromise = import('z-ai-web-dev-sdk')
      .then((m) => (m.default ? m.default.create() : m.create()))
      .catch((e) => {
        zaiPromise = null;
        throw e;
      });
  }
  return zaiPromise;
}

function aiAvailable() {
  if (AI_PROVIDER === 'openai') return Boolean(OPENAI_KEY);
  return true; // SDK провайдер: ключи берёт из окружения платформы, проверяется при первом вызове
}

function isRetriableAiError(err) {
  const msg = String((err && err.message) || err || '');
  return msg.startsWith('ai_upstream_')
    || msg === 'ai_timeout'
    || msg === 'ai_empty'
    || /fetch failed|network|ECONN|EAI_AGAIN|socket|aborted/i.test(msg);
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// AI-вызов с ретраями: временные сбои сети/модели не роняют запрос (итого до 3 попыток)
async function aiChat(messages, { temperature = 0.7, timeoutMs = 90000, attempts = 3 } = {}) {
  const attempt = async () => {
    const run = async () => {
      if (AI_PROVIDER === 'openai') {
        const r = await fetch(OPENAI_BASE + '/chat/completions', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: 'Bearer ' + OPENAI_KEY,
          },
          body: JSON.stringify({ model: OPENAI_MODEL, messages, temperature }),
          signal: AbortSignal.timeout(timeoutMs),
        });
        if (!r.ok) throw new Error('ai_upstream_' + r.status);
        const j = await r.json();
        return j.choices?.[0]?.message?.content || '';
      }
      const zai = await getZai();
      const completion = await zai.chat.completions.create({
        messages,
        thinking: { type: 'disabled' },
      });
      return completion?.choices?.[0]?.message?.content || '';
    };
    let timer;
    const timeout = new Promise((_, reject) => {
      timer = setTimeout(() => reject(new Error('ai_timeout')), timeoutMs);
    });
    try {
      return await Promise.race([run(), timeout]);
    } finally {
      clearTimeout(timer);
    }
  };
  let lastErr;
  for (let i = 1; i <= Math.max(1, attempts); i++) {
    try {
      const text = await attempt();
      if (!String(text).trim()) throw new Error('ai_empty');
      return text;
    } catch (e) {
      lastErr = e;
      if (i === attempts || !isRetriableAiError(e)) break;
      await sleep(500 * i);
    }
  }
  throw lastErr;
}

// Генерация структурированного JSON (quiz/plan/cards): до 2 попыток, повтор — с более строгой инструкцией
async function generateJson({ prompt, validate, temperature = 0.8, timeoutMs = 75000 }) {
  let lastErr;
  for (let i = 0; i < 2; i++) {
    try {
      const raw = await aiChat(
        [{ role: 'assistant', content: BASE_SYS }, { role: 'user', content: prompt + (i ? '\n\nВАЖНО: предыдущая попытка не прошла проверку. Верни строго валидный JSON — без markdown-обёрток, без текста до и после.' : '') }],
        { temperature, timeoutMs, attempts: 2 }
      );
      return validate(extractJson(raw));
    } catch (e) {
      lastErr = e;
      if (e?.message === 'ai_timeout') break; // долгий таймаут не имеет смысла повторять целиком
    }
  }
  throw lastErr;
}

// ---------- системные промпты ----------
const BASE_SYS = `Ты — StudyFlow AI, умный и дружелюбный ИИ-репетитор образовательной платформы StudyFlow.
Ты помогаешь школьникам и студентам: математика, физика, химия, биология, история, география, английский язык, информатика, русский язык, литература, обществознание.
Правила:
- Отвечай на русском языке (если ученик пишет на другом — отвечай на его языке).
- Объясняй понятно и структурированно; сложные темы — простыми словами и с примерами из жизни.
- Задачи решай пошагово: каждый шаг с коротким пояснением. В конце делай проверку ответа.
- Учитывай уровень ученика: если он просит «проще» — упрощай, если просит подробнее — углубляй.
- Используй умеренный markdown: **жирный** для ключевого, списки, код в \`...\`. Формулы — обычным текстом (x², √, π, 10³).
- Не выдумывай факты, даты и формулы. Если не уверен — честно скажи об этом.
- Отвлекаться на темы не про учёбу можно коротко, но всегда возвращай разговор к учёбе.`;

const DEPTH_SYS = {
  brief: 'Стиль ответа: кратко, только суть, до ~120 слов.',
  standard: 'Стиль ответа: средней длины, с примерами и пояснениями.',
  expert: `РЕЖИМ ЭКСПЕРТА (Premium): давай максимально подробный разбор — несколько способов решения,
типичные ошибки по этой теме, 2-3 дополнительных примера и 1-2 вопроса ученику для самопроверки.`,
};

function quizPrompt({ subject, topic, difficulty, count }) {
  return `Сгенерируй школьный тест и верни ТОЛЬКО валидный JSON без markdown-обёрток, без пояснений до и после.
Формат строго такой:
{"questions":[{"q":"текст вопроса","options":["вариант1","вариант2","вариант3","вариант4"],"answer":0,"explain":"краткое объяснение, 1-2 предложения"}]}
Предмет: ${subject}
Тема: ${topic}
Сложность: ${difficulty} из 3
Количество вопросов: ровно ${count}
Требования: ровно 4 варианта ответа в каждом вопросе; ровно один правильный; answer — индекс правильного (0..3);
варианты правдоподобные и одинаковой природы; не повторяющиеся вопросы; вопросы идут по возрастанию сложности;
explain объясняет, ПОЧЕМУ ответ верный. Все тексты на русском.`;
}

function planPrompt({ goal, deadline, days, level }) {
  return `Составь персональный план обучения и верни ТОЛЬКО валидный JSON без markdown-обёрток.
Формат строго такой:
{"tasks":[{"day":1,"title":"короткая задача","detail":"что конкретно сделать, 1-2 предложения"}]}
Цель ученика: ${goal}
Уровень: ${level}
Дедлайн: ${deadline || 'не указан'}; на подготовку примерно ${days} дн.
Требования: разбей цель на ${Math.max(4, Math.min(20, Math.round(days * 1.5)))} небольших конкретных задач;
каждая задача выполнима за 30-90 минут; равномерно распредели по дням (1..${days}); задачи идут от простого к сложному;
в конце — повторение и самопроверка. Все тексты на русском.`;
}

function cardsPrompt({ topic, count }) {
  return `Создай учебные flashcards и верни ТОЛЬКО валидный JSON без markdown-обёрток.
Формат строго такой:
{"cards":[{"f":"вопрос или термин (коротко)","b":"ответ или определение (1-2 предложения)"}]}
Тема: ${topic}
Количество карточек: ровно ${count}
Требования: карточки покрывают тему с разных сторон (определения, формулы, факты, примеры); тексты на русском; без повторов.`;
}

// ---------- надёжное извлечение JSON из ответа модели ----------
function extractJson(text) {
  if (typeof text !== 'string' || !text.trim()) throw new Error('empty');
  let t = text.trim().replace(/```(?:json)?/gi, '').trim();
  const a = t.indexOf('{');
  const b = t.lastIndexOf('}');
  if (a === -1 || b === -1 || b <= a) throw new Error('no_json');
  return JSON.parse(t.slice(a, b + 1));
}

function asSafeStr(v, maxLen) {
  if (typeof v !== 'string') return null;
  const s = v.replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f]/g, '').trim();
  if (!s || s.length > maxLen) return null;
  return s;
}

function validateQuestions(arr, count) {
  if (!Array.isArray(arr)) throw new Error('questions_not_array');
  const out = [];
  for (const q of arr.slice(0, count)) {
    const text = asSafeStr(q?.q, 500);
    const opts = Array.isArray(q?.options) ? q.options.slice(0, 4).map((o) => asSafeStr(o, 200)) : null;
    const ans = Number(q?.answer);
    const exp = asSafeStr(q?.explain, 600) || '';
    if (!text || !opts || opts.length !== 4 || opts.some((o) => o === null)) continue;
    if (!Number.isInteger(ans) || ans < 0 || ans > 3) continue;
    out.push({ q: text, options: opts, answer: ans, explain: exp });
  }
  if (!out.length) throw new Error('no_valid_questions');
  return out;
}

function validateTasks(obj, maxTasks) {
  const arr = Array.isArray(obj) ? obj : Array.isArray(obj?.tasks) ? obj.tasks : null;
  if (!arr) throw new Error('tasks_not_array');
  const out = [];
  for (const t of arr.slice(0, maxTasks)) {
    const title = asSafeStr(t?.title, 200);
    const detail = asSafeStr(t?.detail, 500) || '';
    const day = Math.max(1, Math.min(365, Number(t?.day) || 1));
    if (!title) continue;
    out.push({ day, title, detail });
  }
  if (!out.length) throw new Error('no_valid_tasks');
  return out;
}

function validateCards(obj, count) {
  const arr = Array.isArray(obj) ? obj : Array.isArray(obj?.cards) ? obj.cards : null;
  if (!arr) throw new Error('cards_not_array');
  const out = [];
  for (const c of arr.slice(0, count)) {
    const f = asSafeStr(c?.f, 300);
    const b = asSafeStr(c?.b, 600);
    if (!f || !b) continue;
    out.push({ f, b });
  }
  if (!out.length) throw new Error('no_valid_cards');
  return out;
}

// ---------- сессии (анонимные, без личных данных) ----------
const sessions = new Map();

function todayKey() {
  const d = new Date();
  return d.getFullYear() + '-' + String(d.getMonth() + 1).padStart(2, '0') + '-' + String(d.getDate()).padStart(2, '0');
}

function newSession() {
  const token = crypto.randomBytes(24).toString('hex');
  const s = { created: Date.now(), day: todayKey(), plan: 'free', premiumSince: null, usage: { chat: 0, quiz: 0, plan: 0, cards: 0 } };
  sessions.set(token, s);
  saveSessionsSoon();
  return { token, s };
}

function getSession(req) {
  const token = String(req.get('x-session-token') || '');
  const s = token && sessions.get(token);
  if (!s) return null;
  if (s.day !== todayKey()) {
    s.day = todayKey();
    s.usage = { chat: 0, quiz: 0, plan: 0, cards: 0 };
  }
  return { token, s };
}

// sessions.json: сохраняет тарифы между перезапусками (для продакшена заменить на БД)
let saveTimer = null;
function saveSessionsSoon() {
  if (saveTimer) return;
  saveTimer = setTimeout(() => {
    saveTimer = null;
    try {
      const obj = {};
      for (const [k, v] of sessions) obj[k] = v;
      fs.writeFileSync(SESSIONS_FILE, JSON.stringify({ day: todayKey(), sessions: obj }, null, 1));
    } catch (e) {
      console.error('sessions save failed:', e.message);
    }
  }, 500);
}

try {
  if (fs.existsSync(SESSIONS_FILE)) {
    const data = JSON.parse(fs.readFileSync(SESSIONS_FILE, 'utf8'));
    for (const [k, v] of Object.entries(data.sessions || {})) {
      if (v && typeof v === 'object' && v.usage) sessions.set(k, v);
    }
  }
} catch (e) {
  console.error('sessions load failed:', e.message);
}

function quotaOf(sess) {
  const lim = LIMITS[sess.s.plan === 'premium' ? 'premium' : 'free'];
  return {
    plan: sess.s.plan,
    chat: { used: sess.s.usage.chat, limit: lim.chat },
    quiz: { used: sess.s.usage.quiz, limit: lim.quiz },
    planq: { used: sess.s.usage.plan, limit: lim.plan },
    cards: { used: sess.s.usage.cards, limit: lim.cards },
  };
}

function checkQuota(sess, kind) {
  const q = quotaOf(sess)[kind === 'plan' ? 'planq' : kind];
  if (q.used >= q.limit) {
    const err = new Error('limit');
    err.payload = { error: 'limit', kind, used: q.used, limit: q.limit, plan: sess.s.plan };
    err.status = 429;
    throw err;
  }
}

function requireSession(req) {
  const sess = getSession(req);
  if (!sess) {
    const err = new Error('no_session');
    err.status = 401;
    err.payload = { error: 'no_session' };
    throw err;
  }
  return sess;
}

function requirePremium(sess, feature) {
  if (sess.s.plan !== 'premium') {
    const err = new Error('premium');
    err.status = 403;
    err.payload = { error: 'premium', feature };
    throw err;
  }
}

// ---------- rate limiting (скользящее окно по IP) ----------
const hits = new Map();
function rateLimit(maxPerMin) {
  return (req, res, next) => {
    const ip = req.ip || req.socket?.remoteAddress || 'unknown';
    const now = Date.now();
    let arr = hits.get(ip);
    if (!arr) arr = [];
    arr = arr.filter((t) => now - t < 60000);
    if (arr.length >= maxPerMin) {
      hits.set(ip, arr);
      return res.status(429).json({ error: 'rate', retryAfterSec: 60 });
    }
    arr.push(now);
    hits.set(ip, arr);
    if (hits.size > 5000) hits.clear(); // защита от роста памяти
    next();
  };
}

// ---------- приложение ----------
const app = express();
app.disable('x-powered-by');
app.set('trust proxy', 1);
app.use(express.json({ limit: '64kb' }));

if (ALLOW_CORS) {
  app.use((req, res, next) => {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, X-Session-Token');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    if (req.method === 'OPTIONS') return res.status(204).end();
    next();
  });
}

app.get('/api/health', (req, res) => res.json({ ok: true }));

app.get('/api/config', (req, res) => {
  res.json({
    ok: true,
    ai: { available: aiAvailable(), provider: AI_PROVIDER },
    limits: LIMITS,
    premiumCodeEnabled: Boolean(PREMIUM_CODE),
  });
});

app.post('/api/session', rateLimit(RATE_PER_MIN), (req, res) => {
  const { token } = newSession();
  res.json({ ok: true, token });
});

app.get('/api/quota', rateLimit(RATE_PER_MIN), (req, res) => {
  const sess = requireSession(req);
  res.json({ ok: true, quota: quotaOf(sess) });
});

// --- Premium по промокоду. Точка расширения: платёжный вебхук вызовет ту же логику выдачи премиума.
app.post('/api/premium/activate', rateLimit(RATE_PER_MIN), (req, res) => {
  const sess = requireSession(req);
  if (!PREMIUM_CODE) return res.status(503).json({ error: 'no_promo_config' });
  const code = asSafeStr(req.body?.code, 100) || '';
  const h = (s) => crypto.createHash('sha256').update(s.trim().toUpperCase()).digest();
  if (!code || !crypto.timingSafeEqual(h(code), h(PREMIUM_CODE))) {
    return res.status(403).json({ error: 'bad_code' });
  }
  sess.s.plan = 'premium';
  if (!sess.s.premiumSince) sess.s.premiumSince = new Date().toISOString();
  saveSessionsSoon();
  res.json({ ok: true, quota: quotaOf(sess) });
});

// --- AI чат ---
app.post('/api/chat', rateLimit(RATE_PER_MIN), rateLimit(AI_RATE_PER_MIN), async (req, res, next) => {
  try {
    const sess = requireSession(req);
    checkQuota(sess, 'chat');
    const msgs = Array.isArray(req.body?.messages) ? req.body.messages : null;
    if (!msgs || !msgs.length) throw badRequest('messages required');
    const depth = ['brief', 'standard', 'expert'].includes(req.body?.depth) ? req.body.depth : 'standard';
    if (depth === 'expert') requirePremium(sess, 'expert');

    const history = [];
    for (const m of msgs.slice(-14)) {
      const role = m?.role === 'user' ? 'user' : m?.role === 'assistant' ? 'assistant' : null;
      const content = role && asSafeStr(typeof m?.content === 'string' ? m.content : '', 8000);
      if (role && content) history.push({ role, content });
    }
    if (!history.length || history[history.length - 1].role !== 'user') throw badRequest('last message must be from user');

    const sys = [BASE_SYS, DEPTH_SYS[depth] || DEPTH_SYS.standard].join('\n\n');
    const text = await aiChat([{ role: 'assistant', content: sys }, ...history]);
    if (!text.trim()) throw new Error('ai_empty');

    sess.s.usage.chat += 1;
    saveSessionsSoon();
    res.json({ ok: true, content: text, quota: quotaOf(sess) });
  } catch (e) {
    next(e);
  }
});

// --- Генератор тестов ---
app.post('/api/quiz', rateLimit(RATE_PER_MIN), rateLimit(AI_RATE_PER_MIN), async (req, res, next) => {
  try {
    const sess = requireSession(req);
    checkQuota(sess, 'quiz');
    const subject = asSafeStr(req.body?.subject, 60);
    const topic = asSafeStr(req.body?.topic, 120);
    const difficulty = Math.max(1, Math.min(3, Number(req.body?.difficulty) || 2));
    const count = Math.max(3, Math.min(15, Number(req.body?.count) || 5));
    if (!subject || !topic) throw badRequest('subject and topic required');

    const questions = await generateJson({
      prompt: quizPrompt({ subject, topic, difficulty, count }),
      validate: (obj) => validateQuestions(obj.questions, count),
      temperature: 0.8,
    });

    sess.s.usage.quiz += 1;
    saveSessionsSoon();
    res.json({ ok: true, questions, quota: quotaOf(sess) });
  } catch (e) {
    next(e);
  }
});

// --- ИИ-план обучения (только Premium) ---
app.post('/api/plan', rateLimit(RATE_PER_MIN), rateLimit(AI_RATE_PER_MIN), async (req, res, next) => {
  try {
    const sess = requireSession(req);
    requirePremium(sess, 'plan');
    checkQuota(sess, 'plan');
    const goal = asSafeStr(req.body?.goal, 300);
    const level = asSafeStr(req.body?.level, 60) || 'средний';
    const deadline = asSafeStr(req.body?.deadline, 40) || '';
    if (!goal) throw badRequest('goal required');
    const days = Math.max(2, Math.min(120, Number(req.body?.days) || 7));

    const tasks = await generateJson({
      prompt: planPrompt({ goal, deadline, days, level }),
      validate: (obj) => validateTasks(obj, Math.round(days * 2)),
      temperature: 0.6,
    });

    sess.s.usage.plan += 1;
    saveSessionsSoon();
    res.json({ ok: true, tasks, quota: quotaOf(sess) });
  } catch (e) {
    next(e);
  }
});

// --- ИИ-генерация карточек (только Premium) ---
app.post('/api/cards', rateLimit(RATE_PER_MIN), rateLimit(AI_RATE_PER_MIN), async (req, res, next) => {
  try {
    const sess = requireSession(req);
    requirePremium(sess, 'cards');
    checkQuota(sess, 'cards');
    const topic = asSafeStr(req.body?.topic, 120);
    if (!topic) throw badRequest('topic required');
    const count = Math.max(3, Math.min(20, Number(req.body?.count) || 8));

    const cards = await generateJson({
      prompt: cardsPrompt({ topic, count }),
      validate: (obj) => validateCards(obj, count),
      temperature: 0.7,
    });

    sess.s.usage.cards += 1;
    saveSessionsSoon();
    res.json({ ok: true, cards, quota: quotaOf(sess) });
  } catch (e) {
    next(e);
  }
});

function badRequest(detail) {
  const err = new Error('bad_request');
  err.status = 400;
  err.payload = { error: 'bad_request', detail };
  return err;
}

// --- статика: отдаём фронтенд (../index.html) ---
app.get('/', (req, res) => {
  const f = path.join(__dirname, '..', 'index.html');
  if (fs.existsSync(f)) return res.sendFile(f);
  res.status(404).send('index.html not found — положите фронтенд рядом с backend/ (см. README)');
});

// --- единая обработка ошибок: наружу не утекают внутренности ---
app.use((req, res) => res.status(404).json({ error: 'not_found' }));
app.use((err, req, res, next) => { // eslint-disable-line no-unused-vars
  if (err?.payload) return res.status(err.status || 400).json(err.payload);
  if (err?.type === 'entity.parse.failed' || err?.type === 'entity.too.large') {
    return res.status(400).json({ error: 'bad_request', detail: 'invalid JSON body' });
  }
  const msg = String(err?.message || '');
  console.error('[api error]', msg);
  if (msg.startsWith('ai_')) return res.status(502).json({ error: msg });
  if (msg === 'empty' || msg === 'no_json' || msg.startsWith('no_valid_') || msg.endsWith('_not_array')) {
    return res.status(502).json({ error: 'ai_bad_response' });
  }
  res.status(500).json({ error: 'server_error' });
});

app.listen(PORT, () => {
  console.log(`StudyFlow AI backend: http://localhost:${PORT}  (AI provider: ${AI_PROVIDER}, premium code: ${PREMIUM_CODE ? 'set' : 'off'})`);
});
