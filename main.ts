// main.ts — Backend "Kịch bản tranh biện 3 phút" | You Can Club
// Deno Deploy. Không database, không lưu dữ liệu người dùng.
//
// Endpoint:
//   POST /api/motions  đề tài -> 4 chủ đề
//   POST /api/custom   chủ đề người dùng tự viết -> chủ đề đã chỉnh + tóm tắt hai phe
//   POST /api/script   chủ đề + phe + giọng -> kịch bản
//   GET  /health
//
// Biến môi trường:
//   GEMINI_API_KEY   bắt buộc
//   ALLOWED_ORIGINS  danh sách origin cách nhau bởi dấu phẩy, mặc định "https://youcanclub.github.io"
//   MODEL_MOTIONS    mặc định "gemini-flash-lite-latest"
//   MODEL_SCRIPT     mặc định "gemini-flash-latest"

const GEMINI_API_KEY = Deno.env.get("GEMINI_API_KEY");
const MODEL_MOTIONS = Deno.env.get("MODEL_MOTIONS") || "gemini-flash-lite-latest";
const MODEL_SCRIPT = Deno.env.get("MODEL_SCRIPT") || "gemini-flash-latest";
const ALLOWED_ORIGINS = (Deno.env.get("ALLOWED_ORIGINS") || "https://youcanclub.github.io")
  .split(",").map((s) => s.trim()).filter(Boolean);

const API_BASE = "https://generativelanguage.googleapis.com/v1beta/models";
const GEMINI_TIMEOUT_MS = 45_000;

// Nhịp trình bày khoảng 175 âm tiết/phút => bài 3 phút cần khoảng 525 âm tiết.
const TARGET_MIN = 490;
const TARGET_MAX = 580;
const ACCEPT_MIN = 440;
const ACCEPT_MAX = 640;

/* ============================================================
   CORS
   ============================================================ */
function corsHeaders(req: Request): HeadersInit {
  const origin = req.headers.get("origin") || "";
  const allow = ALLOWED_ORIGINS.includes(origin) ? origin : ALLOWED_ORIGINS[0];
  return {
    "Access-Control-Allow-Origin": allow,
    "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Max-Age": "86400",
    "Vary": "Origin",
  };
}

function json(req: Request, body: unknown, status = 200, extra: HeadersInit = {}): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json; charset=utf-8", ...corsHeaders(req), ...extra },
  });
}

/* ============================================================
   GIỚI HẠN TẦN SUẤT
   ============================================================ */
const RATE_LIMIT = 10; // lượt / phút / IP
const rateMap = new Map<string, number[]>();
let lastSweep = Date.now();

function sweep(now: number) {
  if (now - lastSweep < 120_000) return;
  lastSweep = now;
  for (const [ip, hits] of rateMap) {
    const live = hits.filter((t) => t > now - 60_000);
    if (live.length) rateMap.set(ip, live);
    else rateMap.delete(ip);
  }
}

function isRateLimited(ip: string): boolean {
  const now = Date.now();
  sweep(now);
  const hits = (rateMap.get(ip) || []).filter((t) => t > now - 60_000);
  hits.push(now);
  rateMap.set(ip, hits);
  return hits.length > RATE_LIMIT;
}

function getIp(req: Request): string {
  return req.headers.get("x-forwarded-for")?.split(",")[0].trim() ||
    req.headers.get("cf-connecting-ip") || "unknown";
}

/* ============================================================
   GỌI GEMINI
   ============================================================ */
class GeminiError extends Error {}
class BlockedError extends Error {}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function callGemini(
  model: string,
  prompt: string,
  schema: unknown,
  temperature: number,
  maxOutputTokens: number,
  // deno-lint-ignore no-explicit-any
): Promise<any> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}/${model}:generateContent?key=${GEMINI_API_KEY}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: AbortSignal.timeout(GEMINI_TIMEOUT_MS),
      body: JSON.stringify({
        contents: [{ role: "user", parts: [{ text: prompt }] }],
        generationConfig: {
          temperature,
          topP: 0.95,
          maxOutputTokens,
          responseMimeType: "application/json",
          responseSchema: schema,
        },
        // Nới ngưỡng cho đề tài gai góc (chính trị học đường, mâu thuẫn gia đình...),
        // riêng nội dung khiêu dâm giữ ngưỡng chặt vì người dùng là học sinh THPT.
        safetySettings: [
          { category: "HARM_CATEGORY_HARASSMENT", threshold: "BLOCK_ONLY_HIGH" },
          { category: "HARM_CATEGORY_HATE_SPEECH", threshold: "BLOCK_ONLY_HIGH" },
          { category: "HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold: "BLOCK_MEDIUM_AND_ABOVE" },
          { category: "HARM_CATEGORY_DANGEROUS_CONTENT", threshold: "BLOCK_ONLY_HIGH" },
        ],
      }),
    });
  } catch (e) {
    throw new GeminiError(`Không gọi được Gemini: ${(e as Error).name}`);
  }

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new GeminiError(`Gemini trả mã ${res.status}: ${text.slice(0, 300)}`);
  }

  const data = await res.json();
  if (data.promptFeedback?.blockReason) throw new BlockedError("Đề bài bị bộ lọc an toàn từ chối.");

  const cand = data.candidates?.[0];
  if (!cand) throw new GeminiError("Gemini không trả về ứng viên nào.");
  if (cand.finishReason === "SAFETY" || cand.finishReason === "PROHIBITED_CONTENT") {
    throw new BlockedError("Nội dung bị bộ lọc an toàn từ chối.");
  }
  if (cand.finishReason === "MAX_TOKENS") throw new GeminiError("Gemini bị cắt giữa chừng vì chạm trần token.");

  const raw = cand.content?.parts?.map((p: { text?: string }) => p.text || "").join("") || "";
  if (!raw.trim()) throw new GeminiError("Gemini trả về rỗng.");

  try {
    return JSON.parse(raw.replace(/^```(?:json)?|```$/g, "").trim());
  } catch {
    throw new GeminiError("Gemini trả về JSON không đọc được.");
  }
}

async function callWithRetry(
  model: string,
  prompt: string,
  schema: unknown,
  temperature: number,
  maxOutputTokens: number,
  // deno-lint-ignore no-explicit-any
): Promise<any> {
  try {
    return await callGemini(model, prompt, schema, temperature, maxOutputTokens);
  } catch (err) {
    if (err instanceof BlockedError) throw err;
    console.warn("Gọi lại sau lỗi:", (err as Error).message);
    await sleep(600);
    return await callGemini(model, prompt, schema, temperature, maxOutputTokens);
  }
}

/* ============================================================
   XỬ LÝ VĂN BẢN
   ============================================================ */
const syllables = (t: string) => (t.trim().match(/\S+/g) || []).length;

// Gột những thứ model hay chèn thêm dù đã bị cấm trong prompt.
function tidy(text: string): string {
  return String(text || "")
    .replace(/\r/g, "")
    .replace(/```[\s\S]*?```/g, "")
    .replace(/\*\*|\*|__|`|^#{1,6}\s+/gm, "")
    .replace(/\[[^\]\n]{0,80}\]/g, "")
    .replace(/\(\s*\d{1,2}:\d{2}[^)]*\)/g, "")
    .replace(/\d{1,2}:\d{2}\s*[-–—]\s*\d{1,2}:\d{2}/g, "")
    .replace(/^\s*(?:Luận điểm|Đoạn|Phần|Mở đầu|Kết luận)\s*\d*\s*[:.\-–]\s*/gim, "")
    .replace(/[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}\u{FE0F}]/gu, "")
    .replace(/[ \t]{2,}/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

// Dấu hiệu bịa số liệu: phần trăm và "theo nghiên cứu".
const FAKE_STAT = /\d{1,3}\s?%|\b(theo|dựa trên)\s+(một\s+)?(nghiên cứu|khảo sát|thống kê|báo cáo|số liệu)/i;

// Dấu hiệu AI tưởng tượng một khán phòng trang trọng thay vì buổi sinh hoạt CLB.
const FORMAL_ADDRESS =
  /ban giám khảo|hội đồng|quý vị|kính thưa|thưa (các )?(thầy|cô|anh|chị)|kính mong|trân trọng|xin phép (được )?trình bày|em xin|chúng em xin|các em học sinh|thế hệ trẻ|giới trẻ (ngày nay|hiện nay)|xã hội (ngày nay|hiện đại)|chúng tôi (nhận định|cho rằng|kính)|nhận định rằng|đảm đương trọng trách|trọng trách|thiết nghĩ|có thể thấy rằng|như đã (nêu|trình bày|phân tích) ở trên|tóm lại,? có thể nói/i;

// Mở bài sáo rỗng kiểu văn mẫu.
const ESSAY_OPENERS =
  /^(trong (xã hội|cuộc sống|thời đại) (ngày nay|hiện nay|hiện đại)|từ (xưa )?đến nay|như chúng ta đã biết|có (thể|lẽ) (ai trong chúng ta )?cũng)/i;

// Lưới an toàn cuối: thay thẳng bằng regex để người dùng không bao giờ thấy
// "hội đồng", "ban giám khảo"... lọt ra màn hình, kể cả khi AI sai cả hai lượt.
const FORMAL_REPLACEMENTS: [RegExp, string][] = [
  [/ban giám khảo/gi, "các bạn"],
  [/hội đồng/gi, "các bạn"],
  [/quý vị/gi, "các bạn"],
  [/kính thưa[^,.\n]{0,40}/gi, ""],
  [/kính mong/gi, "mong"],
  [/trân trọng/gi, ""],
  [/xin phép (được )?trình bày/gi, "mình xin chia sẻ"],
  [/chúng em xin/gi, "mình xin"],
  [/\bem xin\b/gi, "mình xin"],
  [/các em học sinh/gi, "các bạn"],
  [/thế hệ trẻ/gi, "tụi mình"],
  [/giới trẻ (ngày nay|hiện nay)/gi, "học sinh bây giờ"],
  [/chúng tôi (nhận định|cho rằng)/gi, "mình nghĩ"],
  [/nhận định rằng/gi, "nghĩ rằng"],
  [/đảm đương trọng trách/gi, "chịu trách nhiệm"],
  [/trọng trách/gi, "trách nhiệm"],
  [/thiết nghĩ/gi, "mình nghĩ"],
  // Người nghe không cần biết đây là bài "3 phút".
  [/\btrong\s+(3|ba)\s+phút(\s+này)?\b/gi, "trong bài nói này"],
  [/\b(3|ba)\s+phút\b/gi, "vài phút"],
];

function neutralizeFormal(text: string): string {
  let out = String(text || "");
  for (const [re, rep] of FORMAL_REPLACEMENTS) out = out.replace(re, rep);
  return out
    .replace(/[ \t]{2,}/g, " ")
    .replace(/\s+([,.!?])/g, "$1")
    .replace(/^[,\s]+/, "")
    .trim();
}

/* ============================================================
   /api/motions — đề tài -> 4 chủ đề
   ============================================================ */
const MOTIONS_SCHEMA = {
  type: "object",
  properties: {
    status: { type: "string", enum: ["ok", "invalid_topic"] },
    message: { type: "string" },
    motions: {
      type: "array",
      items: {
        type: "object",
        properties: {
          motion: { type: "string" },
          pro_summary: { type: "string" },
          con_summary: { type: "string" },
        },
        required: ["motion", "pro_summary", "con_summary"],
        propertyOrdering: ["motion", "pro_summary", "con_summary"],
      },
    },
  },
  required: ["status", "message", "motions"],
  propertyOrdering: ["status", "message", "motions"],
};

// Quy tắc chung cho mọi tóm tắt phe, dùng ở cả /api/motions và /api/custom.
const SUMMARY_RULES = `Mỗi tóm tắt là một câu 18 đến 32 âm tiết, nêu LÝ DO cốt lõi của phe đó, không nhắc lại
nội dung chủ đề. Hai câu phải va vào đúng một điểm tranh cãi và cân sức nhau. Viết như một học
sinh nói về bạn bè mình, không viết như người lớn nói về "các em". Viết "Học sinh tự chịu trách
nhiệm với lựa chọn của mình thì mới thật sự trưởng thành", không viết "Phe ủng hộ cho rằng nên
để các em tự quyết".`;

const AGE_RULE = `Được phép động tới các vấn đề xã hội có tranh cãi thật, kể cả hơi gai góc, miễn còn phù hợp
lứa tuổi THPT. Chỉ tránh nội dung khiêu dâm, kích động bạo lực nghiêm trọng, thù ghét nhắm vào
một nhóm người cụ thể, hoặc cổ suý trực tiếp hành vi phạm pháp nghiêm trọng.`;

function motionsPrompt(topic: string, avoid: string[], idea: string): string {
  const avoidBlock = avoid.length
    ? `\nĐÃ DÙNG RỒI, KHÔNG LẶP Ý\n${avoid.map((m) => `- ${m}`).join("\n")}\n`
    : "";
  const ideaBlock = idea
    ? `\nĐỊNH HƯỚNG RIÊNG NGƯỜI DÙNG GÕ THÊM (tham khảo, không bắt buộc theo tuyệt đối)\n${idea}\nNếu định hướng này hợp lý thì lồng nó vào ít nhất một trong bốn chủ đề; nếu nó mơ hồ hoặc khó tranh biện thì cứ tự chọn góc tốt hơn.\n`
    : "";

  return `VAI TRÒ
Bạn là một bạn học sinh lớp 11 phụ trách mảng tranh biện của You Can Club, một câu lạc bộ toàn
học sinh THPT. Bạn đang nghĩ chủ đề cho buổi sinh hoạt CLB tuần này, không phải một đề thi.

BỐI CẢNH SỬ DỤNG
Một bạn học sinh sẽ đứng nói trước các thành viên khác trong CLB, không có ban giám khảo chấm
điểm. Chủ đề vì vậy nên là chuyện các bạn ấy thật sự va chạm hằng ngày trong đời học sinh,
không phải đề tài học thuật xa vời.

TINH THẦN CHUNG: ƯU TIÊN SÁNG TẠO, ĐỪNG TỰ KIỂM DUYỆT QUÁ TAY
Đây chỉ là bước tạo ý tưởng thô, còn qua một bước rà soát riêng của CLB. Cứ mạnh dạn chọn góc
thú vị, sắc, có chút gai góc nếu nó làm chủ đề hay hơn, đừng chọn phương án an toàn, nhạt nhẽo.

NHIỆM VỤ
Từ đề tài người dùng đưa, tạo đúng 4 chủ đề tranh biện cho bài nói 3 phút.

MỖI CHỦ ĐỀ PHẢI ĐẠT
1. Mở đầu bằng "Chúng tôi tin rằng", "Chúng tôi ủng hộ" hoặc "Chúng tôi phản đối". Nêu rõ ai
   hành động và hành động gì. Một câu, tối đa 28 âm tiết.
2. Hai phe đều có ít nhất hai lý lẽ đứng được. Nếu một người bình thường đọc xong thấy ngay bên
   nào đúng thì bỏ chủ đề đó đi.
3. Thắng được bằng lập luận và ví dụ đời sống học đường, không cần số liệu chuyên ngành.
4. Không dùng từ so sánh mơ hồ nếu chưa nói rõ so với cái gì.
5. ${AGE_RULE}

BỐN CHỦ ĐỀ PHẢI KHÁC NHAU VỀ GÓC, KHÔNG CHỈ KHÁC CÁCH DIỄN ĐẠT
- Một chủ đề về chính sách nhà trường.
- Một chủ đề về trách nhiệm cá nhân của học sinh.
- Một chủ đề về vai trò gia đình hoặc xã hội.
- Một chủ đề phải đánh đổi giữa hai điều cùng tốt.
${avoidBlock}${ideaBlock}
HAI TÓM TẮT PHE
${SUMMARY_RULES}

NẾU ĐỀ TÀI KHÔNG DÙNG ĐƯỢC
status là "invalid_topic", message là một câu tiếng Việt thân thiện nói rõ vì sao và gợi một đề
tài gần đó, motions là mảng rỗng. Chỉ dùng nhánh này khi đề tài THẬT SỰ không thể tranh biện
công bằng được (chỉ có một phía hợp lý, hoặc vi phạm mục 5) — đừng từ chối chỉ vì đề tài nghe
lạ hay hơi nhạy cảm. Nếu dùng được: status là "ok", message là chuỗi rỗng.

Toàn bộ đầu ra bằng tiếng Việt tự nhiên, giọng của một học sinh, không phải giọng người lớn viết
cho học sinh.

ĐỀ TÀI
${topic}`;
}

async function handleMotions(req: Request): Promise<Response> {
  let body: { topic?: string; avoid_motions?: string[]; idea?: string };
  try {
    body = await req.json();
  } catch {
    return json(req, { message: "Dữ liệu gửi lên không đọc được." }, 400);
  }

  const topic = (body.topic || "").trim();
  const avoid = Array.isArray(body.avoid_motions)
    ? body.avoid_motions.filter((m) => typeof m === "string").slice(-12)
    : [];
  const idea = (typeof body.idea === "string" ? body.idea : "").trim().slice(0, 300);

  if (topic.length < 3 || topic.length > 100) {
    return json(req, { message: "Đề tài cần dài từ 3 đến 100 ký tự." }, 400);
  }

  try {
    const result = await callWithRetry(
      MODEL_MOTIONS,
      motionsPrompt(topic, avoid, idea),
      MOTIONS_SCHEMA,
      1.0, // cần đa dạng góc nhìn, nhiệt độ thấp làm 4 chủ đề na ná nhau
      2400,
    );

    if (result.status === "invalid_topic") {
      return json(req, {
        status: "invalid_topic",
        message: result.message || "Đề tài này khó tranh biện, bạn thử một đề tài có hai luồng ý kiến rõ rệt nhé.",
        motions: [],
      });
    }

    const motions = (Array.isArray(result.motions) ? result.motions : [])
      .map((m: Record<string, string>) => ({
        motion: neutralizeFormal(tidy(m.motion)),
        pro_summary: neutralizeFormal(tidy(m.pro_summary)),
        con_summary: neutralizeFormal(tidy(m.con_summary)),
      }))
      .filter((m: Record<string, string>) => m.motion && m.pro_summary && m.con_summary)
      .slice(0, 4);

    if (!motions.length) throw new GeminiError("Không có chủ đề nào hợp lệ.");
    return json(req, { status: "ok", message: "", motions });
  } catch (err) {
    if (err instanceof BlockedError) {
      return json(req, {
        status: "invalid_topic",
        message: "Đề tài này chưa phù hợp để tranh biện trong khuôn khổ câu lạc bộ, bạn thử đề tài khác nhé.",
        motions: [],
      });
    }
    console.error("handleMotions:", err);
    return json(req, { message: "Chưa tạo được chủ đề lúc này, thử lại sau ít phút nhé." }, 502);
  }
}

/* ============================================================
   /api/custom — người dùng tự viết chủ đề
   ============================================================ */
const CUSTOM_SCHEMA = {
  type: "object",
  properties: {
    status: { type: "string", enum: ["ok", "invalid_motion"] },
    message: { type: "string" },
    motion: { type: "string" },
    pro_summary: { type: "string" },
    con_summary: { type: "string" },
  },
  required: ["status", "message", "motion", "pro_summary", "con_summary"],
  propertyOrdering: ["status", "message", "motion", "pro_summary", "con_summary"],
};

function customPrompt(topic: string, motion: string, idea: string): string {
  return `VAI TRÒ
Bạn là một bạn học sinh lớp 11 phụ trách mảng tranh biện của You Can Club. Một bạn trong CLB tự
viết chủ đề để tập nói trước các thành viên khác, không phải thi có giám khảo.

NHIỆM VỤ
1. Chỉnh chủ đề của bạn ấy thành đúng một câu, mở bằng "Chúng tôi tin rằng", "Chúng tôi ủng hộ"
   hoặc "Chúng tôi phản đối", nêu rõ ai hành động và hành động gì, tối đa 28 âm tiết. GIỮ NGUYÊN
   Ý và lập trường gốc, chỉ sửa cho gọn và rõ. Không đổi sang một chủ đề khác.
2. Viết hai tóm tắt phe. ${SUMMARY_RULES}
3. ${AGE_RULE}

NẾU KHÔNG DÙNG ĐƯỢC
Chỉ khi chỉ có một phía hợp lý, hoặc vi phạm mục 3: status là "invalid_motion", message là một
câu tiếng Việt thân thiện nói rõ vì sao và gợi cách viết lại, các trường còn lại là chuỗi rỗng.
Nếu dùng được: status là "ok", message là chuỗi rỗng.

ĐỀ TÀI CHUNG: ${topic || "(không có)"}
${idea ? `ĐỊNH HƯỚNG RIÊNG: ${idea}\n` : ""}
CHỦ ĐỀ NGƯỜI DÙNG VIẾT
${motion}

Toàn bộ đầu ra bằng tiếng Việt tự nhiên, giọng học sinh.`;
}

async function handleCustom(req: Request): Promise<Response> {
  let body: Record<string, string>;
  try {
    body = await req.json();
  } catch {
    return json(req, { message: "Dữ liệu gửi lên không đọc được." }, 400);
  }

  const motion = String(body.motion || "").trim().slice(0, 160);
  const topic = String(body.topic || "").trim().slice(0, 100);
  const idea = String(body.idea || "").trim().slice(0, 300);
  if (motion.length < 10) return json(req, { message: "Chủ đề cần ít nhất 10 ký tự." }, 400);

  try {
    const r = await callWithRetry(MODEL_MOTIONS, customPrompt(topic, motion, idea), CUSTOM_SCHEMA, 0.8, 1200);
    if (r.status === "invalid_motion") {
      return json(req, {
        status: "invalid_motion",
        message: r.message || "Chủ đề này khó tranh biện, bạn thử viết lại cho có hai phía rõ rệt nhé.",
      });
    }
    const m = {
      motion: neutralizeFormal(tidy(r.motion)),
      pro_summary: neutralizeFormal(tidy(r.pro_summary)),
      con_summary: neutralizeFormal(tidy(r.con_summary)),
      custom: true,
    };
    if (!m.motion || !m.pro_summary || !m.con_summary) throw new GeminiError("Thiếu trường.");
    return json(req, { status: "ok", message: "", motion: m });
  } catch (err) {
    if (err instanceof BlockedError) {
      return json(req, {
        status: "invalid_motion",
        message: "Chủ đề này chưa phù hợp để tranh biện trong câu lạc bộ, bạn thử cách viết khác nhé.",
      });
    }
    console.error("handleCustom:", err);
    return json(req, { message: "Chưa xử lý được chủ đề lúc này, thử lại sau ít phút nhé." }, 502);
  }
}

/* ============================================================
   PHONG CÁCH BÀI NÓI
   ============================================================ */
type Style = { label: string; guide: string; temp: number };

const STYLES: Record<string, Style> = {
  "thang-than": {
    label: "Thẳng thắn, rõ ràng",
    temp: 0.75,
    guide: `Vào thẳng vấn đề ngay câu đầu, không rào đón. Câu ngắn, động từ mạnh, hạn chế tính
từ. Mỗi đoạn chốt lại bằng một câu khẳng định gọn. Không dùng ẩn dụ dài dòng, không câu hỏi
tu từ. Sức nặng đến từ lập luận chứ không từ cách nói.`,
  },
  "cam-dong": {
    label: "Kể chuyện, chạm cảm xúc",
    temp: 0.9,
    guide: `Mở đầu bằng một câu chuyện ngắn về một bạn học sinh cụ thể, có tên gọi, có bối cảnh.
Quay lại đúng nhân vật đó ở phần kết để khép vòng. Dùng chi tiết giác quan: tiếng động, ánh
đèn, cái nhìn của người khác. Nhân vật là ví dụ minh hoạ, tuyệt đối không giả vờ đó là người
thật ngoài đời. Cảm xúc đi kèm lập luận, không được thay thế lập luận.`,
  },
  "phan-tich": {
    label: "Điềm tĩnh, phân tích",
    temp: 0.7,
    guide: `Giọng bình thản, chặt chẽ, như người đang gỡ một nút thắt. Ưu tiên quan hệ nhân quả,
nói rõ điều kiện nào thì kết luận đúng và ngoại lệ nằm ở đâu. Dùng cách đặt vấn đề rồi tự trả
lời. Không cảm thán, không hô hào.`,
  },
  "truyen-lua": {
    label: "Truyền lửa, hùng biện",
    temp: 0.9,
    guide: `Nhịp dồn dần từ đầu tới cuối. Dùng điệp ngữ và câu ba vế. Xen câu hỏi tu từ hướng
thẳng vào khán giả. Kết bằng một lời kêu gọi hành động cụ thể, làm được ngay trong trường.
Mỗi câu mạnh phải gắn liền một lý do, tuyệt đối không hô khẩu hiệu suông.`,
  },
  "gan-gui": {
    label: "Gần gũi, hóm hỉnh",
    temp: 0.85,
    guide: `Nói như đang trò chuyện với bạn cùng lớp. Ví dụ lấy từ chuyện thường ngày ở trường.
Được phép pha một câu hóm hỉnh nhẹ nhưng không chế giễu bất kỳ ai và không quá hai lần cả bài.
Tránh từ sách vở. Dù giọng nhẹ, lập luận vẫn phải chặt.`,
  },
};

const DEFAULT_STYLE = "thang-than";
const getStyle = (key?: string): Style => STYLES[key || ""] || STYLES[DEFAULT_STYLE];

// deno-lint-ignore no-explicit-any
function finalizeScript(s: any) {
  return {
    title: neutralizeFormal(s.title),
    sections: s.sections.map((sec: { heading: string; content: string; tip: string }) => ({
      heading: neutralizeFormal(sec.heading),
      content: neutralizeFormal(sec.content),
      tip: neutralizeFormal(sec.tip),
    })),
    rebuttals: (s.rebuttals || []).map((r: { claim: string; response: string }) => ({
      claim: neutralizeFormal(r.claim),
      response: neutralizeFormal(r.response),
    })),
  };
}

/* ============================================================
   /api/script
   ============================================================ */
const SCRIPT_SCHEMA = {
  type: "object",
  properties: {
    title: { type: "string" },
    sections: {
      type: "array",
      items: {
        type: "object",
        properties: {
          heading: { type: "string" },
          content: { type: "string" },
          tip: { type: "string" },
        },
        required: ["heading", "content", "tip"],
        propertyOrdering: ["heading", "content", "tip"],
      },
    },
    rebuttals: {
      type: "array",
      items: {
        type: "object",
        properties: { claim: { type: "string" }, response: { type: "string" } },
        required: ["claim", "response"],
        propertyOrdering: ["claim", "response"],
      },
    },
  },
  required: ["title", "sections", "rebuttals"],
  propertyOrdering: ["title", "sections", "rebuttals"],
};

function scriptPrompt(
  motion: string,
  side: string,
  sideSummary: string,
  opponentSummary: string,
  style: Style,
  fixNote?: string,
  idea = "",
): string {
  const sideLabel = side === "pro" ? "Ủng hộ" : "Phản đối";
  const fixBlock = fixNote ? `\nSỬA LẠI BẢN TRƯỚC\n${fixNote}\n` : "";
  const ideaBlock = idea
    ? `\nĐỊNH HƯỚNG RIÊNG CỦA NGƯỜI NÓI (ưu tiên bám sát: góc khai thác, ví dụ muốn nhắc, điều cần tránh)\n${idea}\n`
    : "";

  return `VAI TRÒ
Bạn là một học sinh lớp 12 dày dạn của You Can Club, được các bạn trong CLB nhờ viết hộ kịch bản
để tập nói. Người đọc bài này là một bạn học sinh khác, đứng nói trước các thành viên còn lại
trong một buổi sinh hoạt CLB bình thường: quây quần trong lớp học hoặc phòng sinh hoạt, không
micro trang trọng, không ban giám khảo, không ai chấm điểm. Người nghe là bạn bè cùng trường,
cùng lứa tuổi, nghe xong sẽ vỗ tay và góp ý thân tình.

ĐỀ BÀI
Chủ đề: ${motion}
Phe của người nói: ${sideLabel}
Hướng lập luận của phe mình: ${sideSummary}
Lập luận mạnh nhất của phe đối diện: ${opponentSummary}
${ideaBlock}
NGÂN SÁCH ĐỘ DÀI, ĐÂY LÀ RÀNG BUỘC CỨNG
Người Việt nói trước lớp ở nhịp khoảng 175 âm tiết một phút, nên bài 3 phút cần tổng
${TARGET_MIN} đến ${TARGET_MAX} âm tiết. Đếm âm tiết là đếm các cụm cách nhau bởi dấu cách.
Chia theo năm phần:
- Mở đầu: 90 đến 110
- Luận điểm một: 135 đến 155
- Luận điểm hai: 135 đến 155
- Đáp lại phe đối diện: 100 đến 120
- Kết luận: 55 đến 75

NĂM PHẦN, GIỮ ĐÚNG THỨ TỰ VÀ CÁCH ĐẶT TIÊU ĐỀ
1. heading "Mở đầu": một hình ảnh hoặc tình huống cụ thể mở màn, định nghĩa một đến hai khái
   niệm then chốt trong chủ đề, rồi tuyên bố lập trường.
2. heading "Luận điểm 1: " cộng một cụm ngắn dưới 8 âm tiết tóm ý.
3. heading "Luận điểm 2: " cộng một cụm ngắn dưới 8 âm tiết tóm ý.
4. heading "Đáp lại ý kiến trái chiều".
5. heading "Kết luận".

MỖI LUẬN ĐIỂM PHẢI ĐỦ BỐN NHỊP
Tuyên bố luận điểm trong một câu. Giải thích cơ chế, vì sao chuyện đó xảy ra. Một ví dụ cụ thể
có người và có tình huống. Cuối cùng nói rõ ai chịu thiệt nếu làm ngược lại.

PHẦN ĐÁP LẠI
Nhắc lại lập luận mạnh nhất của phe đối diện bằng giọng công bằng, thừa nhận phần đúng của nó,
rồi chỉ ra vì sao nó vẫn chưa đủ để lật lập trường của mình.

CẤM BỊA
Không nêu phần trăm, số liệu khảo sát, tên nghiên cứu, tên tổ chức, năm công bố hay trích dẫn
người thật. Thuyết phục bằng lập luận nhân quả và ví dụ ai cũng kiểm chứng được: lớp học, kỳ
thi, bữa cơm gia đình, nhóm chat của lớp, xe buýt, khu trọ, phòng y tế trường, chính sinh hoạt
của CLB.

VĂN PHONG NỀN — ĐÂY LÀ PHẦN QUAN TRỌNG NHẤT
Viết đúng như một bạn học sinh sắp lên nói trước các bạn cùng CLB, không phải một bài văn nghị
luận xã hội được đọc thành tiếng. Bài văn nghị luận thì trang trọng, câu dài, nhiều từ Hán Việt;
bài nói CLB thì như đang trò chuyện, câu ngắn, từ đời thường.

Xưng "mình" (không "tôi", không "chúng tôi" khi nói trực tiếp với khán giả), gọi người nghe là
"các bạn" hoặc "mọi người", gọi phía còn lại là "phía đối diện". Tuyệt đối không dùng bất kỳ
hình thức xưng hô nào coi người nghe là hội đồng chấm điểm hay đám đông xa lạ: không "ban giám
khảo", "hội đồng", "quý vị", "kính thưa", "kính mong", "trân trọng", "xin phép trình bày", "em
xin", "các em học sinh". Người nói và người nghe ngang hàng, đều là học sinh.

Câu dưới 22 âm tiết, mỗi câu một ý. Được phép mở đầu kiểu đang bắt chuyện, nhắc thẳng một chuyện
quen thuộc trong CLB hay trong trường, nhưng đừng lặp kiểu mở này ở nhiều phần. Tuyệt đối không
mở bài hay một đoạn bằng câu sáo mòn: "Trong xã hội ngày nay", "Từ xưa đến nay", "Như chúng ta
đã biết", "Có thể thấy rằng", "Có lẽ ai trong chúng ta". Hạn chế từ Hán Việt nặng và từ hành
chính: viết "trường học chịu trách nhiệm", đừng viết "nhà trường phải đảm đương trọng trách";
viết "mình nghĩ", đừng viết "chúng tôi nhận định rằng"; tránh hẳn "trọng trách", "thiết nghĩ",
"nhận định", "vấn nạn", "thực trạng". Nối câu bằng từ nói miệng: "nhưng mà", "thế nên", "vậy
thì", "với lại", thay vì "tuy nhiên", "bên cạnh đó", "chính vì vậy" lặp đi lặp lại.

VÍ DỤ ĐỐI CHIẾU
Sai: "Kính thưa quý vị, hôm nay em xin trình bày trước hội đồng về vấn nạn áp lực học tập mà thế
hệ trẻ đang gặp phải."
Đúng: "Chắc nhiều bạn ở đây cũng từng thức tới 1 giờ sáng ôn bài, mình cũng vậy, và mình nghĩ
chuyện đó có gì đó sai sai."
Bám sát giọng ở ví dụ "Đúng" cho toàn bộ bài, kể cả khi đổi sang giọng ${style.label}.

Bảo vệ phe ${sideLabel} từ đầu đến cuối, tuyệt đối không kết luận kiểu cả hai bên đều có lý.

GIỌNG NGƯỜI NÓI CHỌN: ${style.label}
${style.guide}
Giọng này phủ lên toàn bài, kể cả title và tip. Nhưng giọng không được phá ngân sách âm tiết,
không được bỏ bốn nhịp của luận điểm, không được vi phạm phần cấm bịa, và vẫn phải giữ cách
xưng hô ngang hàng "mình" — "các bạn".

TUYỆT ĐỐI KHÔNG XUẤT HIỆN TRONG content
Mốc thời gian dưới mọi hình thức. Ngoặc vuông và ghi chú tông giọng. Nhãn đầu đoạn kiểu "Luận
điểm 1:". Markdown, dấu sao, emoji. Ghi chú số âm tiết. Từ hay cách xưng hô trang trọng đã liệt
kê ở trên. Mỗi content là văn bản thuần, ngăn đoạn bằng ký tự xuống dòng.

TRƯỜNG title
Một câu chốt lập trường, 8 đến 14 âm tiết, dùng làm tên bài. Không phải nhan đề chung chung.

TRƯỜNG tip
Mỗi phần kèm một câu ngắn dưới 20 âm tiết mách cách trình bày phần đó: chỗ nào cần chậm lại,
chỗ nào cần nhìn khán giả, chỗ nào cần nhấn giọng.

TRƯỜNG rebuttals
Đúng hai mục. claim là câu phía đối diện nhiều khả năng sẽ hỏi hoặc phản bác, viết như lời nói
thật của một bạn học sinh, không phải văn bản pháp lý. response là cách đáp lại trong hai câu,
dưới 45 âm tiết, giọng vẫn là "mình" nói với "các bạn".${fixBlock}
Toàn bộ đầu ra bằng tiếng Việt tự nhiên, giọng nói miệng của một học sinh THPT.`;
}

// deno-lint-ignore no-explicit-any
function normalizeScript(raw: any) {
  const sections = (Array.isArray(raw.sections) ? raw.sections : [])
    .map((s: Record<string, string>) => ({
      heading: tidy(s.heading),
      content: tidy(s.content),
      tip: tidy(s.tip || ""),
    }))
    .filter((s: Record<string, string>) => s.heading && s.content);

  const rebuttals = (Array.isArray(raw.rebuttals) ? raw.rebuttals : [])
    .map((r: Record<string, string>) => ({ claim: tidy(r.claim), response: tidy(r.response) }))
    .filter((r: Record<string, string>) => r.claim && r.response)
    .slice(0, 3);

  return { title: tidy(raw.title), sections, rebuttals };
}

// deno-lint-ignore no-explicit-any
function totalSyllables(s: any): number {
  return s.sections.reduce((n: number, x: { content: string }) => n + syllables(x.content), 0);
}

async function handleScript(req: Request): Promise<Response> {
  let body: Record<string, string>;
  try {
    body = await req.json();
  } catch {
    return json(req, { message: "Dữ liệu gửi lên không đọc được." }, 400);
  }

  const motion = (body.motion || "").trim();
  const side = body.side;
  const sideSummary = (body.side_summary || "").trim();
  const opponentSummary = (body.opponent_summary || "").trim();
  const idea = String(body.idea || "").trim().slice(0, 300);
  const style = getStyle(body.style); // khoá lạ thì rơi về giọng mặc định

  if (!motion || (side !== "pro" && side !== "con") || !sideSummary || !opponentSummary) {
    return json(req, { message: "Thiếu thông tin chủ đề hoặc phe tranh biện." }, 400);
  }

  try {
    let best = normalizeScript(await callWithRetry(
      MODEL_SCRIPT,
      scriptPrompt(motion, side, sideSummary, opponentSummary, style, undefined, idea),
      SCRIPT_SCHEMA,
      style.temp,
      4000,
    ));

    if (!best.title || best.sections.length < 3) throw new GeminiError("Kịch bản thiếu phần.");

    // Một vòng sửa duy nhất, chỉ khi thật sự cần. Lỗi xưng hô trang trọng không
    // kích hoạt vòng sửa vì finalizeScript đã thay thẳng bằng regex (đỡ tốn 8–20 giây).
    const count = totalSyllables(best);
    const joined = best.sections.map((s: { content: string }) => s.content).join(" ");
    const notes: string[] = [];

    if (count < ACCEPT_MIN) {
      notes.push(`Bản trước chỉ khoảng ${count} âm tiết, quá ngắn so với 3 phút. Viết dài hơn, thêm ví dụ cụ thể, đạt ${TARGET_MIN} đến ${TARGET_MAX} âm tiết.`);
    } else if (count > ACCEPT_MAX) {
      notes.push(`Bản trước khoảng ${count} âm tiết, nói ra sẽ quá 3 phút. Cắt bớt câu thừa, giữ ${TARGET_MIN} đến ${TARGET_MAX} âm tiết.`);
    }
    if (FAKE_STAT.test(joined)) {
      notes.push("Bản trước có số liệu hoặc nghiên cứu không kiểm chứng được. Bỏ hết, thay bằng ví dụ đời sống học đường.");
    }
    if (ESSAY_OPENERS.test(best.sections[0]?.content || "")) {
      notes.push('Câu mở đầu bị sáo mòn kiểu văn mẫu nghị luận. Viết lại phần Mở đầu bằng một tình huống hoặc câu nói cụ thể, đời thường, không dùng các cụm mở bài kiểu "trong xã hội ngày nay" hay "như chúng ta đã biết".');
    }

    if (notes.length) {
      try {
        const retryRaw = normalizeScript(await callGemini(
          MODEL_SCRIPT,
          scriptPrompt(motion, side, sideSummary, opponentSummary, style, notes.join(" "), idea),
          SCRIPT_SCHEMA,
          style.temp,
          4000,
        ));
        // Chỉ thay nếu bản mới thật sự gần mục tiêu hơn.
        const mid = (TARGET_MIN + TARGET_MAX) / 2;
        const better = retryRaw.sections.length >= 3 &&
          Math.abs(totalSyllables(retryRaw) - mid) <= Math.abs(count - mid) + 40;
        if (better) best = retryRaw;
      } catch (e) {
        console.warn("Vòng sửa thất bại, giữ bản đầu:", (e as Error).message);
      }
    }

    best = finalizeScript(best);
    return json(req, { ...best, style: style.label });
  } catch (err) {
    if (err instanceof BlockedError) {
      return json(req, { message: "Chủ đề này không sinh được kịch bản, bạn chọn chủ đề khác nhé." }, 400);
    }
    console.error("handleScript:", err);
    return json(req, { message: "Chưa viết được kịch bản lúc này, thử lại sau ít phút nhé." }, 502);
  }
}

/* ============================================================
   ROUTER
   ============================================================ */
const HANDLERS: Record<string, (r: Request) => Promise<Response>> = {
  "/api/motions": handleMotions,
  "/api/custom": handleCustom,
  "/api/script": handleScript,
};

Deno.serve(async (req: Request) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { status: 204, headers: corsHeaders(req) });
  }

  const url = new URL(req.url);

  if (req.method === "GET" && url.pathname === "/health") {
    return json(req, {
      ok: Boolean(GEMINI_API_KEY),
      models: { motions: MODEL_MOTIONS, script: MODEL_SCRIPT },
      styles: Object.keys(STYLES),
      endpoints: Object.keys(HANDLERS),
    });
  }

  if (!GEMINI_API_KEY) {
    return json(req, { message: "Máy chủ chưa cấu hình GEMINI_API_KEY." }, 500);
  }

  if (req.method === "POST" && HANDLERS[url.pathname]) {
    if (isRateLimited(getIp(req))) {
      return json(req, { message: "Đang có nhiều yêu cầu cùng lúc, đợi một phút rồi thử lại nhé." }, 429, {
        "Retry-After": "60",
      });
    }
    return await HANDLERS[url.pathname](req);
  }

  return json(req, { message: "Không tìm thấy đường dẫn này." }, 404);
});
