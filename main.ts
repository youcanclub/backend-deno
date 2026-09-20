// main.ts — Backend cho "Tạo Kịch Bản Tranh Biện 3 Phút" (You Can Club)
// Chạy trên Deno Deploy. Không dùng database, không lưu trữ gì cả.

const GEMINI_API_KEY = Deno.env.get("GEMINI_API_KEY");
const ALLOWED_ORIGIN = Deno.env.get("ALLOWED_ORIGIN") || "https://youcanclub.github.io";
const GEMINI_MODEL = "gemini-2.5-flash";
const GEMINI_URL =
  `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent`;

// ---------- Rate limit đơn giản trong bộ nhớ (per-instance) ----------
// Đủ dùng cho quy mô 1 CLB. Không chống được abuse phân tán trên nhiều region,
// nhưng chặn được việc 1 người bấm liên tục.
const RATE_LIMIT = 8; // request / phút / IP
const rateMap = new Map<string, number[]>();

function isRateLimited(ip: string): boolean {
  const now = Date.now();
  const windowStart = now - 60_000;
  const hits = (rateMap.get(ip) || []).filter((t) => t > windowStart);
  hits.push(now);
  rateMap.set(ip, hits);
  return hits.length > RATE_LIMIT;
}

function getIp(req: Request): string {
  return req.headers.get("x-forwarded-for")?.split(",")[0].trim() ||
    req.headers.get("cf-connecting-ip") ||
    "unknown";
}

// ---------- CORS ----------
function corsHeaders(): HeadersInit {
  return {
    "Access-Control-Allow-Origin": ALLOWED_ORIGIN,
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
  };
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", ...corsHeaders() },
  });
}

// ---------- Gọi Gemini ----------
async function callGemini(
  prompt: string,
  schema: unknown,
  temperature: number,
  maxOutputTokens: number,
): Promise<any> {
  const res = await fetch(`${GEMINI_URL}?key=${GEMINI_API_KEY}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      contents: [{ role: "user", parts: [{ text: prompt }] }],
      generationConfig: {
        temperature,
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

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new GeminiError(`Gemini API lỗi ${res.status}: ${text.slice(0, 300)}`);
  }

  const data = await res.json();
  const candidate = data.candidates?.[0];

  // Bị chặn bởi safety filter hoặc không sinh ra nội dung
  if (!candidate || candidate.finishReason === "SAFETY") {
    throw new BlockedError("Nội dung bị hệ thống an toàn của Gemini từ chối.");
  }

  const rawText = candidate.content?.parts?.[0]?.text;
  if (!rawText) {
    throw new GeminiError("Gemini không trả về nội dung.");
  }

  try {
    return JSON.parse(rawText);
  } catch {
    throw new GeminiError("Gemini trả về JSON không hợp lệ.");
  }
}

class GeminiError extends Error {}
class BlockedError extends Error {}

async function callGeminiWithRetry(
  prompt: string,
  schema: unknown,
  temperature: number,
  maxOutputTokens: number,
): Promise<any> {
  try {
    return await callGemini(prompt, schema, temperature, maxOutputTokens);
  } catch (err) {
    if (err instanceof BlockedError) throw err; // không retry nếu bị chặn an toàn
    // Retry đúng 1 lần cho lỗi mạng/JSON hỏng/5xx
    return await callGemini(prompt, schema, temperature, maxOutputTokens);
  }
}

// ---------- /api/motions ----------
const MOTIONS_SCHEMA = {
  type: "OBJECT",
  properties: {
    status: { type: "STRING", enum: ["ok", "invalid_topic"] },
    message: { type: "STRING" },
    motions: {
      type: "ARRAY",
      items: {
        type: "OBJECT",
        properties: {
          id: { type: "INTEGER" },
          motion: { type: "STRING" },
          pro_summary: { type: "STRING" },
          con_summary: { type: "STRING" },
        },
        required: ["id", "motion", "pro_summary", "con_summary"],
      },
    },
  },
  required: ["status", "message", "motions"],
};

function buildMotionsPrompt(topic: string, avoidMotions: string[]): string {
  const avoidBlock = avoidMotions.length
    ? `\nTRÁNH LẶP: không tạo kiến nghị trùng ý với các kiến nghị đã đưa trước đó:\n${
      avoidMotions.map((m) => `- ${m}`).join("\n")
    }\n`
    : "";

  return `VAI TRÒ
Bạn là Cố vấn Tranh biện của You Can Club, làm việc với học sinh THPT Việt Nam.

NHIỆM VỤ
Từ "Chủ đề lớn" người dùng nhập, tạo đúng 4 Kiến nghị tranh biện (motions)
phù hợp cho một bài nói 3 phút.

TIÊU CHÍ TỪNG KIẾN NGHỊ
1. Dạng chuẩn: bắt đầu bằng "Chúng tôi tin rằng...", "Chúng tôi sẽ..." hoặc
   "Chúng tôi phản đối...". Phải nêu rõ CHỦ THỂ hành động và HÀNH ĐỘNG cụ thể.
2. Cân bằng: mỗi phe phải có ít nhất 2 lý lẽ hợp lý. Nếu một người hiểu biết
   trung bình thấy ngay "rõ ràng đúng" hoặc "rõ ràng sai" thì loại bỏ.
3. Tranh được trong 3 phút: phạm vi hẹp, thắng bằng lập luận và ví dụ đời sống
   học đường, không đòi hỏi số liệu chuyên ngành.
4. Rõ nghĩa: không dùng từ mơ hồ nếu không kèm tiêu chí so sánh.
5. Chuẩn mực học đường: văn minh, tích cực, không chính trị nhạy cảm, không
   tôn giáo/sắc tộc, không bạo lực, không vi phạm quy định nhà trường.

RÀNG BUỘC ĐA DẠNG
4 kiến nghị phải khác nhau về góc tiếp cận: (1) chính sách, (2) giá trị,
(3) góc nhìn cá nhân/học sinh, (4) góc nhìn nhà trường hoặc xã hội.
${avoidBlock}
TÓM TẮT HAI PHE
Mỗi tóm tắt 1 câu, 20-35 từ, nêu LÝ DO cốt lõi ("vì..."), không nhắc lại nội
dung kiến nghị. pro_summary và con_summary phải va chạm trực diện vào cùng
một điểm tranh cãi.

GIỚI HẠN ĐỘ DÀI
motion: 1 câu, tối đa 30 từ. Toàn bộ đầu ra bằng tiếng Việt tự nhiên.

XỬ LÝ ĐẦU VÀO KHÔNG HỢP LỆ
Nếu chủ đề trống, vô nghĩa, hay vi phạm chuẩn mực học đường: status =
"invalid_topic", message là 1 câu tiếng Việt thân thiện giải thích lý do,
motions là mảng rỗng. Nếu hợp lệ: status = "ok", message = "".

ĐẦU VÀO
Chủ đề lớn: ${topic}`;
}

async function handleMotions(req: Request): Promise<Response> {
  let body: { topic?: string; avoid_motions?: string[] };
  try {
    body = await req.json();
  } catch {
    return json({ message: "Dữ liệu gửi lên không hợp lệ." }, 400);
  }

  const topic = (body.topic || "").trim();
  const avoidMotions = Array.isArray(body.avoid_motions) ? body.avoid_motions.slice(0, 8) : [];

  if (topic.length < 3 || topic.length > 100) {
    return json({ message: "Chủ đề cần dài từ 3 đến 100 ký tự." }, 400);
  }

  try {
    const result = await callGeminiWithRetry(
      buildMotionsPrompt(topic, avoidMotions),
      MOTIONS_SCHEMA,
      0.7,
      2000,
    );
    return json(result);
  } catch (err) {
    if (err instanceof BlockedError) {
      return json({
        status: "invalid_topic",
        message: "Chủ đề này chưa phù hợp để tranh biện trong khuôn khổ CLB, bạn thử chủ đề khác nhé.",
        motions: [],
      });
    }
    console.error("handleMotions error:", err);
    return json({ message: "Không tạo được kiến nghị lúc này, thử lại sau nhé." }, 502);
  }
}

// ---------- /api/script ----------
const SCRIPT_SCHEMA = {
  type: "OBJECT",
  properties: {
    title: { type: "STRING" },
    sections: {
      type: "ARRAY",
      items: {
        type: "OBJECT",
        properties: {
          heading: { type: "STRING" },
          content: { type: "STRING" },
        },
        required: ["heading", "content"],
      },
    },
  },
  required: ["title", "sections"],
};

const FORBIDDEN_PATTERN = /\[|\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2}/;

function buildScriptPrompt(
  motion: string,
  side: string,
  sideSummary: string,
  opponentSummary: string,
  lengthHint?: string,
): string {
  const sideLabel = side === "pro" ? "Ủng hộ (Pro)" : "Phản đối (Con)";
  const lengthBlock = lengthHint ? `\nLƯU Ý ĐỘ DÀI: ${lengthHint}\n` : "";

  return `VAI TRÒ
Bạn là Huấn luyện viên Tranh biện của You Can Club. Viết kịch bản bài nói
tranh biện 3 phút cho học sinh THPT Việt Nam.

ĐẦU VÀO
Kiến nghị tranh biện: ${motion}
Phe của người nói: ${sideLabel}
Luận điểm phe mình: ${sideSummary}
Luận điểm phe đối thủ: ${opponentSummary}

NGUYÊN TẮC
1. Bảo vệ phe ${sideLabel} từ đầu đến cuối. Không trung lập, không kết luận
   kiểu "cả hai phe đều có lý".
2. Dùng luận điểm phe mình làm hạt nhân rồi phát triển đầy đủ, không chép
   nguyên văn.
3. KHÔNG bịa số liệu, tên nghiên cứu, tên tổ chức, năm khảo sát hay trích dẫn
   chuyên gia. Thuyết phục bằng lập luận nhân quả, ví dụ đời sống học đường.

CẤU TRÚC VÀ DUNG LƯỢNG (tổng 380-420 từ)
Đề mục 1 — "Mở đầu & Bối cảnh" (khoảng 80 từ): câu mở gây chú ý, định nghĩa
1-2 khái niệm then chốt, tuyên bố lập trường.
Đề mục 2 — "Luận điểm chính & Phản biện" (khoảng 230 từ): hai luận điểm theo
mạch khẳng định → cơ chế → ví dụ → chốt; sau đó một đoạn nêu lại lập luận
mạnh nhất của phe đối thủ dựa trên luận điểm đối thủ rồi bác bỏ.
Đề mục 3 — "Kết luận & Khẳng định" (khoảng 80 từ): tóm tắt bằng cách diễn đạt
mới, nâng lên giá trị lớn hơn, kết bằng câu mạnh nhắc lại kiến nghị.
Giữ nguyên chính xác ba tên đề mục trên.${lengthBlock}
VĂN PHONG
Viết để NÓI. Câu ngắn (dưới 25 từ), chủ động. Xưng "chúng tôi", gọi đối thủ
là "phe đối diện". Từ nối rõ: "Thứ nhất", "Ngược lại", "Chính vì vậy". Đanh
thép, quả quyết, tôn trọng. Tiếng Việt tự nhiên, tránh văn dịch máy.

TUYỆT ĐỐI KHÔNG XUẤT HIỆN TRONG "content"
Mốc thời gian dưới mọi hình thức; ngoặc vuông [ ] và ghi chú hành động/tông
giọng; nhãn đầu dòng kiểu "Luận điểm 1:"; Markdown; emoji; lời chào ban giám
khảo; ghi chú số từ; lời giải thích ngoài kịch bản.

Mỗi "content" là một chuỗi văn bản thuần, ngăn đoạn bằng ký tự xuống dòng.`;
}

function countWords(sections: { content: string }[]): number {
  return sections.map((s) => s.content).join(" ").trim().split(/\s+/).filter(Boolean).length;
}

async function handleScript(req: Request): Promise<Response> {
  let body: {
    motion?: string;
    side?: string;
    side_summary?: string;
    opponent_summary?: string;
  };
  try {
    body = await req.json();
  } catch {
    return json({ message: "Dữ liệu gửi lên không hợp lệ." }, 400);
  }

  const { motion, side, side_summary, opponent_summary } = body;
  if (!motion || (side !== "pro" && side !== "con") || !side_summary || !opponent_summary) {
    return json({ message: "Thiếu thông tin kiến nghị hoặc phe tranh biện." }, 400);
  }

  try {
    let result = await callGeminiWithRetry(
      buildScriptPrompt(motion, side, side_summary, opponent_summary),
      SCRIPT_SCHEMA,
      0.3,
      1200,
    );

    // Tự sửa 1 lần nếu độ dài lệch khoảng cho phép
    let wordCount = countWords(result.sections);
    if (wordCount < 340 || wordCount > 460) {
      const hint = wordCount < 340
        ? `Bản trước chỉ có khoảng ${wordCount} từ, hãy viết dài hơn, đủ 380-420 từ.`
        : `Bản trước có khoảng ${wordCount} từ, hãy viết ngắn lại, đủ 380-420 từ.`;
      result = await callGemini(
        buildScriptPrompt(motion, side, side_summary, opponent_summary, hint),
        SCRIPT_SCHEMA,
        0.3,
        1200,
      );
    }

    // Tự sửa 1 lần nếu còn sót mốc giờ / ngoặc vuông
    const joined = result.sections.map((s: { content: string }) => s.content).join(" ");
    if (FORBIDDEN_PATTERN.test(joined)) {
      result = await callGemini(
        buildScriptPrompt(
          motion,
          side,
          side_summary,
          opponent_summary,
          "Bản trước còn sót mốc thời gian hoặc dấu ngoặc vuông, hãy loại bỏ hoàn toàn.",
        ),
        SCRIPT_SCHEMA,
        0.3,
        1200,
      );
    }

    return json(result);
  } catch (err) {
    if (err instanceof BlockedError) {
      return json({ message: "Nội dung này không thể sinh kịch bản, thử kiến nghị khác nhé." }, 400);
    }
    console.error("handleScript error:", err);
    return json({ message: "Không tạo được kịch bản lúc này, thử lại sau nhé." }, 502);
  }
}

// ---------- Router ----------
Deno.serve(async (req: Request) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders() });
  }

  if (!GEMINI_API_KEY) {
    return json({ message: "Server thiếu cấu hình GEMINI_API_KEY." }, 500);
  }

  const ip = getIp(req);
  if (isRateLimited(ip)) {
    return json({ message: "Quá nhiều yêu cầu, đợi một phút rồi thử lại." }, 429);
  }

  const url = new URL(req.url);

  if (req.method === "POST" && url.pathname === "/api/motions") {
    return handleMotions(req);
  }
  if (req.method === "POST" && url.pathname === "/api/script") {
    return handleScript(req);
  }

  return json({ message: "Không tìm thấy endpoint." }, 404);
});
