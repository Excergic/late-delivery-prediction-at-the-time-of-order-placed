import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL;
const API_KEY = process.env.API_KEY ?? "";

export async function POST(req: NextRequest) {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (API_KEY) {
    headers["Authorization"] = `Bearer ${API_KEY}`;
  }

  const res = await fetch(`${BACKEND_URL}/predict`, {
    method: "POST",
    headers,
    body: JSON.stringify(await req.json()),
  });

  const data = await res.json();
  return NextResponse.json(data, { status: res.status });
}
