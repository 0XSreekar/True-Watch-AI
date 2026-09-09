/**
 * Demo fixtures for TRUE WATCH.
 * Everything in this file is sample data — replace with real repositories/queries
 * when the database and the detection pipeline are wired up.
 */

export const CAMERAS = [
  { id: 'RXL-01', name: 'BOP Raxaul — Pillar 42', sector: 'Raxaul', ir: false, road: true },
  { id: 'JGB-03', name: 'Jogbani Ridge — Trail Head', sector: 'Jogbani', ir: true, road: false },
  { id: 'PNT-07', name: 'Panitanki — River Bank', sector: 'Panitanki', ir: true, road: true },
  { id: 'SNL-01', name: 'Sunauli Gate — Lane 2', sector: 'Sunauli', ir: false, road: true },
  { id: 'GLG-05', name: 'Galgalia — Culvert Approach', sector: 'Galgalia', ir: true, road: false },
  { id: 'PHU-02', name: 'Phuentsholing Line — Post 9', sector: 'Phuentsholing', ir: false, road: true },
];

export const POSTS = [
  { id: 'RXL', name: 'Raxaul', x: 17, y: 63, road: true, alert: false },
  { id: 'BRG', name: 'Birgunj Culvert', x: 31, y: 43, road: true, alert: false },
  { id: 'JGB', name: 'Jogbani Ridge', x: 45, y: 71, road: false, alert: true },
  { id: 'GLG', name: 'Galgalia', x: 58, y: 37, road: false, alert: false },
  { id: 'PNT', name: 'Panitanki', x: 71, y: 61, road: true, alert: false },
  { id: 'PHU', name: 'Phuentsholing', x: 85, y: 41, road: true, alert: false },
  { id: 'SNL', name: 'Sunauli', x: 8, y: 33, road: true, alert: false },
];

/** Written by the VLM judge in the real system. */
export const REASONS = [
  'Loaded vehicle moving north at 02:41, outside sanctioned hours, on a route with no recorded night traffic.',
  'Group of six crossing the dry channel 140 m east of Pillar 42, single file, no lamps, holding formation.',
  'Pickup halted 40 s at the fence line, two figures transferred bundles to the far side, then reversed out.',
  'Motorcycle on a Nepali plate re-entered the same lane a fourth time in 22 minutes, carrying no load on return.',
  'Person tracked climbing the embankment inside the virtual fence at Galgalia culvert, moving against footfall.',
];

/** ANPR samples — Latin and Devanagari series. */
export const PLATES = [
  'BR 06 GA 4821',
  'UP 32 CN 9013',
  'बा १२ च ४५६७',
  'WB 73 AB 1290',
  'प्र १ ख २३४५',
  'BR 22 PA 7745',
];

export const SEARCH_PLACEHOLDERS = [
  'white pickup near pillar 42, night of the 14th',
  'group of more than four people after midnight, Jogbani',
  'motorcycle with Nepali plate, Panitanki, last 10 days',
  'anyone carrying a long bundle across the dry channel',
  'vehicles that stopped at the fence for over 30 seconds',
];

export const SEARCH_CHIPS = [
  ['white', 'pickup', 'northbound'],
  ['group', '6 persons', 'no lamps'],
  ['motorcycle', 'NP plate', 'return trip'],
  ['bundle', 'carried', 'dry channel'],
  ['halted 42 s', 'fence line'],
  ['single person', 'IR', 'embankment'],
];

export const UNITS = [
  'Sector HQ — Patna Frontier',
  '20th Bn — Raxaul',
  '36th Bn — Jogbani',
  '12th Bn — Panitanki',
  '41st Bn — Sunauli',
  '55th Bn — Phuentsholing',
];

export const ROLES = [
  { k: 'Operator', dd: 'Watch wall, act on alerts' },
  { k: 'Supervisor', dd: 'Approve, unmask faces, tune fences' },
  { k: 'HQ Analyst', dd: 'Cross-sector search and reporting' },
];

/** Alarm budget per operator shift — the gate that keeps alerts credible. */
export const SHIFT_ALARM_BUDGET = 12;

export const ANALYTICS = {
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

export const OPERATOR = {
  name: 'Ct. R. K. Thapa',
  initials: 'RT',
  unit: '20th Bn Raxaul',
  defaultRole: 'Operator',
};
