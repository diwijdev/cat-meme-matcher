export async function matchFace(features) {
  const res = await fetch("http://localhost:8000/match", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(features),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || "API request failed");
  }

  return await res.json(); // {label, conf}
}
