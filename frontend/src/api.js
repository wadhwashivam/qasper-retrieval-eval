const BASE = import.meta.env.VITE_API_URL;

async function request(path, options) {
    let res;
    try {
        res = await fetch(`${BASE}${path}`, options);
    } catch {
        throw new Error("Can't reach the API. Is uvicorn running on port 8000?");
    }
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
        const detail = typeof data.detail === "string" ? data.detail : "Invalid input.";
        throw new Error(detail);
    }
    return data;
}

export const getPapers = () => request("/papers");
export const getQuestions = (id) => request(`/papers/${encodeURIComponent(id)}/questions`);
export const ask = (paper_id, question) =>
    request("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ paper_id, question }),
    });