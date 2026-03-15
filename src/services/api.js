/**
 * Centralised API client.
 * All paths are relative so Vite's /api proxy (→ http://localhost:8006) is used.
 */

const BASE = import.meta.env.VITE_API_URL ?? '/api';

function authHeaders() {
  const token = localStorage.getItem('token');
  const headers = { 'Content-Type': 'application/json' };
  if (token) headers['Authorization'] = `Bearer ${token}`;
  return headers;
}

async function request(method, path, body) {
  const res = await fetch(`${BASE}${path}`, {
    method,
    headers: authHeaders(),
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });

  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.detail || `Request failed (${res.status})`);
  }
  return data;
}

export const login = (email, password) =>
  request('POST', '/auth/login', { email, password });

export const signup = (email, password, full_name) =>
  request('POST', '/auth/signup', { email, password, full_name });

export const getMe = () =>
  request('GET', '/auth/me');

/**
 * Generate a mood-conditioned arpeggio.
 *
 * @param {Object} params
 * @param {string}  params.key         - Musical key, e.g. "C"
 * @param {string}  params.scale       - Scale type, e.g. "major"
 * @param {number}  params.tempo       - BPM (20–400)
 * @param {number}  params.note_count  - Number of notes (1–128)
 * @param {string}  params.mood        - Free-text mood description
 * @param {number}  params.octave      - Starting octave (0–8)
 * @param {string}  params.pattern     - Arpeggio pattern
 * @param {number}  [params.seed]      - Optional random seed
 * @returns {Promise<GenerateArpeggioResponse>}
 */
export const generateArpeggio = (params) =>
  request('POST', '/generate-arpeggio', params);

export const saveFavorite = (payload) =>
  request('POST', '/favorites', payload);

export const getFavorites = () =>
  request('GET', '/favorites');

export const deleteFavorite = (id) =>
  request('DELETE', `/favorites/${id}`);

/**
 * Download a saved MIDI file, triggering a browser save-as dialog.
 * Uses fetch directly so we can attach the auth token and handle binary data.
 */
export async function downloadFavorite(id, filename) {
  const token = localStorage.getItem('token');
  const res = await fetch(`${BASE}/favorites/${id}/download`, {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });
  if (!res.ok) throw new Error(`Download failed (${res.status})`);
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
