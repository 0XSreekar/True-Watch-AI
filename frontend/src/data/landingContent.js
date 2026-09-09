/** Copy and figures for the public site. Move to a CMS/API when there is one. */

export const PROBLEMS = [
  {
    fig: 'FIG. 2',
    tag: 'OPEN BORDER',
    accent: '#1F6F7A',
    title: 'Everyone crossing is within their rights',
    body: '2,450 km of open frontier with Nepal and Bhutan. Thousands cross legally every day, so the signal is not movement — it is intent, load, hour and route.',
    kind: 'open',
  },
  {
    fig: 'FIG. 3',
    tag: 'NO ROAD, NO GRID',
    accent: '#B57324',
    title: '308 posts a vehicle cannot reach',
    body: 'Of 734 border out posts, 308 have no road access and intermittent power. Anything that needs a fat uplink or a server room is already out.',
    kind: 'nopower',
  },
  {
    fig: 'FIG. 4',
    tag: 'UNPROVABLE',
    accent: '#6E4B7C',
    title: 'Footage nobody can prove is untouched',
    body: 'Conventional CCTV records to a disk anyone can edit. In court, a clip with no chain of custody is a story, not evidence.',
    kind: 'chain',
  },
];

export const STATS = [
  { target: 2450, suffix: 'KM', label: 'Open border guarded across Nepal and Bhutan' },
  { target: 734, suffix: 'BOP', label: 'Border out posts in the SSB network' },
  { target: 308, suffix: '', label: 'Posts with no road access at all' },
  { target: 8, suffix: '', label: 'Capabilities delivered purely in software' },
  { target: 0, suffix: 'LICENCE', label: 'Runs on cameras already procured', prefix: '₹' },
];

export const PIPELINE = [
  {
    title: 'Camera',
    meta: 'EXISTING CCTV',
    accent: 'rgba(43,37,33,0.66)',
    dot: '#2B2521',
    body: 'Whatever is already on the pole — analogue or IP, day or IR. No replacement, no re-cabling.',
  },
  {
    title: 'Appearance channel',
    meta: 'NEURAL DETECTOR',
    accent: '#3E5169',
    dot: '#3E5169',
    body: 'A quantised detector reads what the object looks like: person, vehicle, load, plate, face region.',
  },
  {
    title: 'Motion channel',
    meta: 'FLOW + PHYSICS',
    accent: '#1F6F7A',
    dot: '#1F6F7A',
    body: 'Optical flow with a physical motion model. It does not care about appearance — only trajectory, speed and mass.',
  },
  {
    title: 'Agreement gate',
    meta: 'BOTH OR NOTHING',
    accent: '#1F6F7A',
    dot: '#1F6F7A',
    body: 'Nothing leaves this gate unless both channels independently agree. This is what holds the alarm budget.',
  },
  {
    title: 'Provisional alert',
    meta: '~30 ms',
    accent: '#B57324',
    dot: '#B57324',
    body: 'A box on the operator wall almost instantly, so a guard can look before the reasoning finishes.',
  },
  {
    title: 'VLM judge',
    meta: 'ON-DEVICE',
    accent: '#8A6A22',
    dot: '#A9812F',
    body: 'A small vision-language model reads the evidence in context — hour, route, load, prior traffic — and writes the reason in plain words.',
  },
  {
    title: 'Confirmed verdict',
    meta: '~3 s',
    accent: '#A32E24',
    dot: '#A32E24',
    body: 'A card with a written justification and a confidence figure the operator can argue with.',
  },
  {
    title: 'Sealed evidence',
    meta: 'SHA-256 CHAIN',
    accent: '#6E4B7C',
    dot: '#6E4B7C',
    body: 'The clip is hashed at capture and chained to the previous block, so tampering is detectable forever.',
  },
];

export const CAPABILITIES = [
  {
    code: 'C-01',
    glyph: '◎',
    title: 'Human detection & tracking',
    desc: 'Persistent track IDs across occlusion, plus group-size counting at the crossing line.',
    a: '#3E5169',
  },
  {
    code: 'C-02',
    glyph: '▤',
    title: 'Vehicle detection & classification',
    desc: 'Truck, pickup, tractor, motorcycle — with a loaded-versus-empty judgement.',
    a: '#1F6F7A',
  },
  {
    code: 'C-03',
    glyph: '◍',
    title: 'Face detection',
    desc: 'Face regions isolated and blurred by default; revealed only under a supervisor warrant.',
    a: '#6E4B7C',
  },
  {
    code: 'C-04',
    glyph: '⌸',
    title: 'ANPR — Latin & Devanagari',
    desc: 'Reads Indian and Nepali plates, including बा १२ च ४५६७ style Devanagari series.',
    a: '#A9812F',
  },
  {
    code: 'C-05',
    glyph: '⬚',
    title: 'Virtual fence',
    desc: 'Operator-drawn polygons per camera with direction-of-crossing rules and hour windows.',
    a: '#B57324',
  },
  {
    code: 'C-06',
    glyph: '△',
    title: 'Suspicious activity',
    desc: 'Loitering, hand-off, reverse flow, repeat passes and abandoned-object patterns.',
    a: '#A32E24',
  },
  {
    code: 'C-07',
    glyph: '☾',
    title: 'Night-time movement',
    desc: 'IR-aware thresholds and a night baseline per route, so 02:00 traffic is judged against 02:00 normality.',
    a: '#3E5169',
  },
  {
    code: 'C-08',
    glyph: '⌁',
    title: 'Real-time alerts & event log',
    desc: 'Two-stage alerting into an append-only event log with a full operator audit trail.',
    a: '#1F6F7A',
  },
];

export const CONSOLE_NAV = [
  { g: '▤', t: 'Live wall' },
  { g: '△', t: 'Alert queue' },
  { g: '◈', t: 'Post map' },
  { g: '⌕', t: 'Footage search' },
  { g: '⛓', t: 'Evidence' },
  { g: '◑', t: 'Analytics' },
  { g: '⬚', t: 'Virtual fences' },
  { g: '⚙', t: 'Settings' },
];

/** DORI coverage bands drawn under every live-wall tile. */
export const DORI_BANDS = [
  { k: 'DETECT', c: 'rgba(62,81,105,0.85)', w: 40 },
  { k: 'OBSERVE', c: 'rgba(31,111,122,0.85)', w: 26 },
  { k: 'RECOGNISE', c: 'rgba(169,129,47,0.9)', w: 20 },
  { k: 'IDENTIFY', c: 'rgba(163,46,36,0.9)', w: 14 },
];
