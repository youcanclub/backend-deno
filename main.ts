// main.ts — Backend "Kịch bản tranh biện 3 phút" | You Can Club
// Deno Deploy. Không database, không lưu dữ liệu người dùng.
//
// Biến môi trường:
//   GEMINI_API_KEY   bắt buộc
//   ALLOWED_ORIGINS  danh sách origin cách nhau bởi dấu phẩy
//                    mặc định "https://youcanclub.github.io"
//   MODEL_MOTIONS    mặc định "gemini-flash-lite-latest"  (nhanh, rẻ)
//   MODEL_SCRIPT     mặc định "gemini-flash-latest"       (viết hay hơn)

const GEMINI_API_KEY = Deno.env.get("GEMINI_API_KEY");
const MODEL_MOTIONS = Deno.env.get("MODEL_MOTIONS") || "gemini-flash-lite-latest";
const MODEL_SCRIPT = Deno.env.get("MODEL_SCRIPT") || "gemini-flash-latest";
const ALLOWED_ORIGINS = (Deno.env.get("ALLOWED_ORIGINS") || "https://youcanclub.github.io")
  .split(",").map((s) => s.trim()).filter(Boolean);

const API_BASE = "https://generativelanguage.googleapis.com/v1beta/models";
const GEMINI_TIMEOUT_MS = 45_000;

// Tiếng Việt nói ở nhịp trình bày khoảng 175 âm tiết mỗi phút.
// Bài 3 phút vì vậy cần khoảng 525 âm tiết, không phải 400.
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
const RATE_LIMIT = 10;              // lượt / phút / IP
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
        safetySettings: [
          { category: "HARM_CATEGORY_HARASSMENT", threshold: "BLOCK_MEDIUM_AND_ABOVE" },
          { category: "HARM_CATEGORY_HATE_SPEECH", threshold: "BLOCK_MEDIUM_AND_ABOVE" },
          { category: "HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold: "BLOCK_MEDIUM_AND_ABOVE" },
          { category: "HARM_CATEGORY_DANGEROUS_CONTENT", threshold: "BLOCK_MEDIUM_AND_ABOVE" },
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

  if (data.promptFeedback?.blockReason) {
    throw new BlockedError("Đề bài bị bộ lọc an toàn từ chối.");
  }

  const cand = data.candidates?.[0];
  if (!cand) throw new GeminiError("Gemini không trả về ứng viên nào.");
  if (cand.finishReason === "SAFETY" || cand.finishReason === "PROHIBITED_CONTENT") {
    throw new BlockedError("Nội dung bị bộ lọc an toàn từ chối.");
  }
  if (cand.finishReason === "MAX_TOKENS") {
    throw new GeminiError("Gemini bị cắt giữa chừng vì chạm trần token.");
  }

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

// Dấu hiệu bịa số liệu. Con số trong ví dụ đời thường thì không sao,
// nhưng phần trăm và "theo nghiên cứu" thì gần như chắc chắn là bịa.
const FAKE_STAT = /\d{1,3}\s?%|\b(theo|dựa trên)\s+(một\s+)?(nghiên cứu|khảo sát|thống kê|báo cáo|số liệu)/i;
const FORMAL_ADDRESS = /ban giám khảo|hội đồng|quý vị|kính thưa|thưa (các )?(thầy|cô|anh|chị)/i;

/* ============================================================
   /api/motions
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

function motionsPrompt(topic: string, avoid: string[]): string {
  const avoidBlock = avoid.length
    ? `\nĐÃ DÙNG RỒI, KHÔNG LẶP Ý\n${avoid.map((m) => `- ${m}`).join("\n")}\n`
    : "";

  return `VAI TRÒ
Bạn là cố vấn tranh biện của You Can Club, một câu lạc bộ toàn học sinh THPT.

BỐI CẢNH SỬ DỤNG
Kiến nghị này dùng cho một buổi sinh hoạt CLB, nơi một bạn học sinh đứng nói trước các
thành viên khác trong CLB, không phải một kỳ thi hùng biện có ban giám khảo chấm điểm.
Kiến nghị vì vậy nên là chuyện các bạn ấy thật sự va chạm hằng ngày trong đời học sinh,
không phải đề tài học thuật xa vời.

NHIỆM VỤ
Từ chủ đề người dùng đưa, tạo đúng 4 kiến nghị tranh biện cho bài nói 3 phút.

MỖI KIẾN NGHỊ PHẢI ĐẠT
1. Mở đầu bằng "Chúng tôi tin rằng", "Chúng tôi ủng hộ" hoặc "Chúng tôi phản đối".
   Nêu rõ ai hành động và hành động gì. Một câu, tối đa 28 âm tiết.
2. Hai phe đều có ít nhất hai lý lẽ đứng được. Nếu một người bình thường đọc xong
   thấy ngay bên nào đúng thì bỏ kiến nghị đó đi.
3. Thắng được bằng lập luận và ví dụ đời sống học đường, không cần số liệu chuyên ngành.
4. Không dùng từ so sánh mơ hồ nếu chưa nói rõ so với cái gì.
5. Phù hợp môi trường học đường: không chính trị nhạy cảm, không tôn giáo, không sắc tộc,
   không bạo lực, không cổ vũ vi phạm nội quy.

BỐN KIẾN NGHỊ PHẢI KHÁC NHAU VỀ GÓC, KHÔNG CHỈ KHÁC CÁCH DIỄN ĐẠT
- Một kiến nghị về chính sách nhà trường.
- Một kiến nghị về trách nhiệm cá nhân của học sinh.
- Một kiến nghị về vai trò gia đình hoặc xã hội.
- Một kiến nghị phải đánh đổi giữa hai điều cùng tốt.
${avoidBlock}
HAI TÓM TẮT PHE
Mỗi tóm tắt là một câu 18 đến 32 âm tiết, nêu LÝ DO cốt lõi của phe đó, không nhắc lại
nội dung kiến nghị. Hai câu phải va vào đúng một điểm tranh cãi và cân sức nhau. Viết như
một học sinh nói về bạn bè mình, không viết như người lớn nói về "các em". Viết "Học sinh
tự chịu trách nhiệm với lựa chọn của mình thì mới thật sự trưởng thành", không viết "Phe
ủng hộ cho rằng nên để các em tự quyết".

NẾU CHỦ ĐỀ KHÔNG DÙNG ĐƯỢC
status là "invalid_topic", message là một câu tiếng Việt thân thiện nói rõ vì sao và gợi
một chủ đề gần đó, motions là mảng rỗng.
Nếu chủ đề dùng được: status là "ok", message là chuỗi rỗng.

Toàn bộ đầu ra bằng tiếng Việt tự nhiên.

CHỦ ĐỀ
${topic}`;
}

async function handleMotions(req: Request): Promise<Response> {
  let body: { topic?: string; avoid_motions?: string[] };
  try {
    body = await req.json();
  } catch {
    return json(req, { message: "Dữ liệu gửi lên không đọc được." }, 400);
  }

  const topic = (body.topic || "").trim();
  const avoid = Array.isArray(body.avoid_motions)
    ? body.avoid_motions.filter((m) => typeof m === "string").slice(-12)
    : [];

  if (topic.length < 3 || topic.length > 100) {
    return json(req, { message: "Chủ đề cần dài từ 3 đến 100 ký tự." }, 400);
  }

  try {
    const result = await callWithRetry(
      MODEL_MOTIONS,
      motionsPrompt(topic, avoid),
      MOTIONS_SCHEMA,
      1.0,          // cần đa dạng góc nhìn, nhiệt độ thấp làm 4 kiến nghị na ná nhau
      2400,
    );

    if (result.status === "invalid_topic") {
      return json(req, {
        status: "invalid_topic",
        message: result.message || "Chủ đề này khó tranh biện, bạn thử một chủ đề có hai luồng ý kiến rõ rệt nhé.",
        motions: [],
      });
    }

    const motions = (Array.isArray(result.motions) ? result.motions : [])
      .map((m: Record<string, string>) => ({
        motion: tidy(m.motion),
        pro_summary: tidy(m.pro_summary),
        con_summary: tidy(m.con_summary),
      }))
      .filter((m: Record<string, string>) => m.motion && m.pro_summary && m.con_summary)
      .slice(0, 4);

    if (!motions.length) throw new GeminiError("Không có kiến nghị nào hợp lệ.");

    return json(req, { status: "ok", message: "", motions });
  } catch (err) {
    if (err instanceof BlockedError) {
      return json(req, {
        status: "invalid_topic",
        message: "Chủ đề này chưa phù hợp để tranh biện trong khuôn khổ câu lạc bộ, bạn thử chủ đề khác nhé.",
        motions: [],
      });
    }
    console.error("handleMotions:", err);
    return json(req, { message: "Chưa tạo được kiến nghị lúc này, thử lại sau ít phút nhé." }, 502);
  }
}

/* ============================================================
   PHONG CÁCH BÀI NÓI
   ============================================================ */
type Style = { label: string; guide: string; temp: number };

const STYLES: Record<string, Style> = {
  "thang-than": {
    label: "Thẳng thắn, rõ ràng",
    temp: 0.7,
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
    temp: 0.65,
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
        properties: {
          claim: { type: "string" },
          response: { type: "string" },
        },
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
): string {
  const sideLabel = side === "pro" ? "Ủng hộ" : "Phản đối";
  const fixBlock = fixNote ? `\nSỬA LẠI BẢN TRƯỚC\n${fixNote}\n` : "";

  return `VAI TRÒ
Bạn là huấn luyện viên tranh biện của You Can Club, một câu lạc bộ toàn học sinh THPT.
Viết kịch bản một bài nói 3 phút. Người đọc là một bạn học sinh, đứng nói trước các bạn
thành viên khác trong buổi sinh hoạt CLB, không phải trước ban giám khảo một cuộc thi.
Người nghe là bạn bè cùng trường, cùng lứa tuổi, không phải người lạ hay người lớn.

ĐỀ BÀI
Kiến nghị: ${motion}
Phe của người nói: ${sideLabel}
Hướng lập luận của phe mình: ${sideSummary}
Lập luận mạnh nhất của phe đối diện: ${opponentSummary}

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
1. heading "Mở đầu": một hình ảnh hoặc tình huống cụ thể mở màn, định nghĩa một đến hai
   khái niệm then chốt trong kiến nghị, rồi tuyên bố lập trường.
2. heading "Luận điểm 1: " cộng một cụm ngắn dưới 8 âm tiết tóm ý.
3. heading "Luận điểm 2: " cộng một cụm ngắn dưới 8 âm tiết tóm ý.
4. heading "Đáp lại ý kiến trái chiều".
5. heading "Kết luận".

MỖI LUẬN ĐIỂM PHẢI ĐỦ BỐN NHỊP
Tuyên bố luận điểm trong một câu. Giải thích cơ chế, vì sao chuyện đó xảy ra. Một ví dụ
cụ thể có người và có tình huống. Cuối cùng nói rõ ai chịu thiệt nếu làm ngược lại.

PHẦN ĐÁP LẠI
Nhắc lại lập luận mạnh nhất của phe đối diện bằng giọng công bằng, thừa nhận phần đúng
của nó, rồi chỉ ra vì sao nó vẫn chưa đủ để lật lập trường của mình.

CẤM BỊA
Không nêu phần trăm, số liệu khảo sát, tên nghiên cứu, tên tổ chức, năm công bố hay trích
dẫn người thật. Thuyết phục bằng lập luận nhân quả và ví dụ ai cũng kiểm chứng được: lớp
học, kỳ thi, bữa cơm gia đình, nhóm chat của lớp, xe buýt, khu trọ, phòng y tế trường,
chính sinh hoạt của CLB.

VĂN PHONG NỀN
Viết để nói, không phải để đọc thầm, và nói với bạn bè chứ không phải tranh tụng trước
toà. Câu dưới 25 âm tiết, mỗi câu một ý. Xưng "mình", gọi người nghe là "các bạn", gọi
phía còn lại là "phía đối diện" thay vì "phe đối diện" cho bớt tính chất đối đầu. Được
phép có một câu mở kiểu đang trò chuyện, ví dụ nhắc thẳng tới một chuyện quen thuộc trong
CLB hay trong trường, miễn không lặp lại giữa các phần. Dùng từ nối rõ ràng. Hạn chế từ
Hán Việt nặng và từ ngữ hành chính: viết "trường học chịu trách nhiệm", đừng viết "nhà
trường phải đảm đương trọng trách"; viết "mình nghĩ", đừng viết "chúng tôi nhận định rằng".
Bảo vệ phe ${sideLabel} từ đầu đến cuối, tuyệt đối không kết luận kiểu cả hai bên đều có lý.

GIỌNG NGƯỜI NÓI CHỌN: ${style.label}
${style.guide}
Giọng này phủ lên toàn bài, kể cả câu title và các câu trong tip. Nhưng giọng không được
phá ngân sách âm tiết, không được bỏ bốn nhịp của luận điểm, và không được vi phạm phần
cấm bịa ở trên.

TUYỆT ĐỐI KHÔNG XUẤT HIỆN TRONG content
Mốc thời gian dưới mọi hình thức. Ngoặc vuông và ghi chú tông giọng. Nhãn đầu đoạn kiểu
"Luận điểm 1:". Markdown, dấu sao, emoji. Ghi chú số âm tiết. Không dùng các từ "ban giám
khảo", "hội đồng", "quý vị", "kính thưa" hay bất kỳ cách xưng hô nào coi người nghe là một
hội đồng chấm điểm — người nghe luôn là các bạn thành viên CLB, không phải ban giám khảo.
Mỗi content là văn bản thuần, ngăn đoạn bằng ký tự xuống dòng.

TRƯỜNG title
Một câu chốt lập trường, 8 đến 14 âm tiết, dùng làm tên bài. Không phải nhan đề chung chung.

TRƯỜNG tip
Mỗi phần kèm một câu ngắn dưới 20 âm tiết mách cách trình bày phần đó: chỗ nào cần chậm
lại, chỗ nào cần nhìn khán giả, chỗ nào cần nhấn giọng.

TRƯỜNG rebuttals
Đúng hai mục. claim là câu phía đối diện nhiều khả năng sẽ hỏi hoặc phản bác, viết như lời
nói thật của một bạn học sinh, không phải văn bản pháp lý. response là cách đáp lại trong
hai câu, dưới 45 âm tiết, giọng vẫn là "mình" nói với "các bạn".${fixBlock}
Toàn bộ đầu ra bằng tiếng Việt tự nhiên.`;
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
  const style = getStyle(body.style);   // khoá lạ thì rơi về giọng mặc định

  if (!motion || (side !== "pro" && side !== "con") || !sideSummary || !opponentSummary) {
    return json(req, { message: "Thiếu thông tin kiến nghị hoặc phe tranh biện." }, 400);
  }

  try {
    let best = normalizeScript(await callWithRetry(
      MODEL_SCRIPT,
      scriptPrompt(motion, side, sideSummary, opponentSummary, style),
      SCRIPT_SCHEMA,
      style.temp,   // 0.3 cho ra văn đúng nhưng nhạt, mỗi giọng có nhiệt độ riêng
      4000,
    ));

    if (!best.title || best.sections.length < 3) {
      throw new GeminiError("Kịch bản thiếu phần.");
    }

    // Một vòng sửa duy nhất, chỉ khi thật sự cần.
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
    if (FORMAL_ADDRESS.test(joined) || FORMAL_ADDRESS.test(best.title)) {
      notes.push('Bản trước xưng hô như đang nói trước ban giám khảo hoặc hội đồng. Đây là buổi sinh hoạt CLB, người nghe là các bạn thành viên. Xưng "mình", gọi người nghe là "các bạn", bỏ hết các từ như "ban giám khảo", "hội đồng", "quý vị", "kính thưa".');
    }

    if (notes.length) {
      try {
        const retryRaw = normalizeScript(await callGemini(
          MODEL_SCRIPT,
          scriptPrompt(motion, side, sideSummary, opponentSummary, style, notes.join(" ")),
          SCRIPT_SCHEMA,
          style.temp,
          4000,
        ));
        // Chỉ thay nếu bản mới thật sự gần mục tiêu hơn.
        const mid = (TARGET_MIN + TARGET_MAX) / 2;
        const better = retryRaw.sections.length >= 3 &&
          Math.abs(totalSyllables(retryRaw) - mid) < Math.abs(count - mid);
        if (better) best = retryRaw;
      } catch (e) {
        console.warn("Vòng sửa thất bại, giữ bản đầu:", (e as Error).message);
      }
    }

    return json(req, { ...best, style: style.label });
  } catch (err) {
    if (err instanceof BlockedError) {
      return json(req, { message: "Kiến nghị này không sinh được kịch bản, bạn chọn kiến nghị khác nhé." }, 400);
    }
    console.error("handleScript:", err);
    return json(req, { message: "Chưa viết được kịch bản lúc này, thử lại sau ít phút nhé." }, 502);
  }
}

/* ============================================================
   ROUTER
   ============================================================ */
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
    });
  }

  if (!GEMINI_API_KEY) {
    return json(req, { message: "Máy chủ chưa cấu hình GEMINI_API_KEY." }, 500);
  }

  if (req.method === "POST" && (url.pathname === "/api/motions" || url.pathname === "/api/script")) {
    if (isRateLimited(getIp(req))) {
      return json(req, { message: "Đang có nhiều yêu cầu cùng lúc, đợi một phút rồi thử lại nhé." }, 429, {
        "Retry-After": "60",
      });
    }
    return url.pathname === "/api/motions" ? handleMotions(req) : handleScript(req);
  }

  return json(req, { message: "Không tìm thấy đường dẫn này." }, 404);
});
