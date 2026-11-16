export default function AskBox({ question, onChange, onAsk, loading }) {
  const canAsk = question.trim().length >= 3 && !loading;

  function handleSubmit(e) {
    e.preventDefault();
    if (canAsk) onAsk();
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) handleSubmit(e);
  }

  return (
    <form onSubmit={handleSubmit}>
      <textarea
        value={question}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKeyDown}
        rows={3}
        maxLength={500}
        placeholder="Ask a question about this paper…"
        className="w-full resize-y rounded-md border border-stone-300 bg-white px-3 py-2 text-sm outline-none focus:border-orange-700"
      />
      <div className="mt-2 flex items-center justify-between">
        <span className="text-xs text-stone-400">⌘ + Enter to ask</span>
        <button
          type="submit"
          disabled={!canAsk}
          className="rounded-md bg-stone-900 px-4 py-2 text-sm font-medium text-white hover:bg-orange-700 disabled:cursor-not-allowed disabled:bg-stone-300"
        >
          {loading ? "Asking…" : "Ask"}
        </button>
      </div>
    </form>
  );
}
