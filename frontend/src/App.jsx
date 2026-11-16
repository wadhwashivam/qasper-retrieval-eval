import { useEffect, useState } from "react";
import { getPapers, getQuestions, ask } from "./api";
import { citedNumbers } from "./citations";
import PaperPicker from "./components/PaperPicker";
import AskBox from "./components/AskBox";
import Answer from "./components/Answer";
import Passages from "./components/Passages";

const DEMO_PAPER = "1912.01214";

export default function App() {
  const [papers, setPapers] = useState([]);
  const [paperId, setPaperId] = useState(DEMO_PAPER);
  const [questions, setQuestions] = useState([]);
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    getPapers()
      .then(setPapers)
      .catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    let ignore = false;
    setResult(null);
    setError("");
    setQuestions([]);
    getQuestions(paperId)
      .then((qs) => {
        if (!ignore) setQuestions(qs);
      })
      .catch((e) => {
        if (!ignore) setError(e.message);
      });
    return () => {
      ignore = true;
    };
  }, [paperId]);

  async function handleAsk() {
    setLoading(true);
    setError("");
    setResult(null);
    try {
      setResult(await ask(paperId, question.trim()));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  const cited = result ? citedNumbers(result.answer) : new Set();
  const title = papers.find((p) => p.id === paperId)?.title;

  return (
    <div className="min-h-screen bg-stone-50 text-stone-900">
      <header className="border-b border-stone-200 bg-white">
        <div className="mx-auto max-w-6xl px-5 py-5">
          <h1 className="text-xl font-semibold tracking-tight">Ask a research paper</h1>
          <p className="mt-1 text-sm text-stone-500">
            Retrieval: paragraph chunks (≤200 tokens) + dense embeddings, top 5 · Generation: Llama 3.1 8B via
            Ollama
          </p>
        </div>
      </header>

      <main className="mx-auto grid max-w-6xl gap-8 px-5 py-8 md:grid-cols-[320px_1fr]">
        <PaperPicker
          papers={papers}
          paperId={paperId}
          onPaperChange={setPaperId}
          questions={questions}
          onPickQuestion={setQuestion}
        />

        <section className="min-w-0">
          {title && (
            <p className="mb-3 text-sm text-stone-500">
              <span className="font-mono text-xs">{paperId}</span> · {title}
            </p>
          )}

          <AskBox question={question} onChange={setQuestion} onAsk={handleAsk} loading={loading} />

          {loading && <p className="mt-6 text-sm text-stone-500">Thinking… this takes a few seconds.</p>}

          {error && (
            <p className="mt-6 rounded-md border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
              {error}
            </p>
          )}

          {result && (
            <>
              <Answer answer={result.answer} />
              <Passages passages={result.passages} cited={cited} />
            </>
          )}
        </section>
      </main>
    </div>
  );
}