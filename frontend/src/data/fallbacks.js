/**
 * Last-resort sample data so the demo still renders if the API is unreachable.
 * The API is the source of truth — these mirror its shapes, nothing more.
 */

export const FALLBACK_CAMERAS = [
  { id: 'RXL-01', name: 'BOP Raxaul — Pillar 42', sector: 'Raxaul', ir: false, road: true },
  { id: 'JGB-03', name: 'Jogbani Ridge — Trail Head', sector: 'Jogbani', ir: true, road: false },
  { id: 'PNT-07', name: 'Panitanki — River Bank', sector: 'Panitanki', ir: true, road: true },
  { id: 'SNL-01', name: 'Sunauli Gate — Lane 2', sector: 'Sunauli', ir: false, road: true },
  { id: 'GLG-05', name: 'Galgalia — Culvert Approach', sector: 'Galgalia', ir: true, road: false },
  { id: 'PHU-02', name: 'Phuentsholing Line — Post 9', sector: 'Phuentsholing', ir: false, road: true },
];

export const FALLBACK_POSTS = [
  { id: 'RXL', name: 'Raxaul', x: 17, y: 63, road: true, alert: false },
  { id: 'BRG', name: 'Birgunj Culvert', x: 31, y: 43, road: true, alert: false },
  { id: 'JGB', name: 'Jogbani Ridge', x: 45, y: 71, road: false, alert: true },
  { id: 'GLG', name: 'Galgalia', x: 58, y: 37, road: false, alert: false },
  { id: 'PNT', name: 'Panitanki', x: 71, y: 61, road: true, alert: false },
  { id: 'PHU', name: 'Phuentsholing', x: 85, y: 41, road: true, alert: false },
  { id: 'SNL', name: 'Sunauli', x: 8, y: 33, road: true, alert: false },
];

export const FALLBACK_PLACEHOLDERS = [
  'white pickup near pillar 42, night of the 14th',
  'group of more than four people after midnight, Jogbani',
  'motorcycle with Nepali plate, Panitanki, last 10 days',
  'anyone carrying a long bundle across the dry channel',
  'vehicles that stopped at the fence for over 30 seconds',
];

export const FALLBACK_UNITS = [
  'Sector HQ — Patna Frontier',
  '20th Bn — Raxaul',
  '36th Bn — Jogbani',
  '12th Bn — Panitanki',
  '41st Bn — Sunauli',
  '55th Bn — Phuentsholing',
];

export const FALLBACK_ROLES = [
  { k: 'Operator', dd: 'Watch wall, act on alerts' },
  { k: 'Supervisor', dd: 'Approve, unmask faces, tune fences' },
  { k: 'HQ Analyst', dd: 'Cross-sector search and reporting' },
];

export const FALLBACK_ANALYTICS = {
  camBars: [
    { id: 'RXL-01', v: 9 },
    { id: 'JGB-03', v: 6 },
    { id: 'PNT-07', v: 11 },
    { id: 'SNL-01', v: 4 },
    { id: 'GLG-05', v: 7 },
    { id: 'PHU-02', v: 5 },
  ],
  dayNight: [
    { k: 'DAY 06–18', v: 38, c: '#B57324' },
    { k: 'NIGHT 18–06', v: 62, c: '#3E5169' },
  ],
  agreement: [91, 93, 92, 95, 94, 96, 95, 97, 96, 97, 98, 97],
  traffic: [4, 6, 5, 8, 7, 9, 6, 11, 8, 7, 9, 12],
};

export const FALLBACK_CHAIN = [
  { n: 1041, hash: 'a71c04e9b2…4f10c8', camera: 'RXL-01', time: '14 AUG · 01:12' },
  { n: 1042, hash: 'c30d5581af…9b2e71', camera: 'JGB-03', time: '14 AUG · 02:19' },
  { n: 1043, hash: '18ba6e2d40…77c1a3', camera: 'PNT-07', time: '14 AUG · 03:26' },
  { n: 1044, hash: 'f92a3b70c5…21de84', camera: 'SNL-01', time: '14 AUG · 04:33' },
  { n: 1045, hash: '4d6e18ca93…b0f527', camera: 'GLG-05', time: '14 AUG · 05:40' },
];
