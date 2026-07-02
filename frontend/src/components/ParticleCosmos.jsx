import { useEffect, useRef } from 'react'

// ─── Point-in-polygon (ray casting) ─────────────────────────────────────────
function pip(lon, lat, poly) {
  let inside = false
  const n = poly.length
  for (let i = 0, j = n - 1; i < n; j = i++) {
    const xi = poly[i][0], yi = poly[i][1]
    const xj = poly[j][0], yj = poly[j][1]
    if (((yi > lat) !== (yj > lat)) &&
        (lon < (xj - xi) * (lat - yi) / (yj - yi) + xi))
      inside = !inside
  }
  return inside
}

// ─── Accurate continent polygon data ─────────────────────────────────────────
// Each poly: [[lon, lat], ...] — simplified but geographically accurate coastlines
const LAND_REGIONS = [

  // ── NORTH AMERICA ──────────────────────────────────────────────────────────
  { id: 'na', color: '#5577ee', glow: '#3355bb', polys: [[
    // CW from Bering Strait → Pacific coast → Gulf → Atlantic coast → Arctic
    [-168,63],[-164,70],[-153,72],[-141,68],[-132,56],[-128,50],
    [-124,48],[-124,44],[-122,37],[-120,34],[-117,32],
    // Baja simplified, then Mexico Pacific
    [-110,24],[-106,20],[-98,18],[-90,16],[-86,14],[-83,10],[-79,8],
    // Caribbean coast back north
    [-76,10],[-82,16],[-87,21],
    // Gulf of Mexico coast (key for recognizability)
    [-90,21],[-95,22],[-97,22],[-97,28],[-97,30],
    [-94,30],[-90,30],[-88,30],[-86,30],[-84,30],[-82,29],[-80,25],
    // Atlantic coast north
    [-80,32],[-77,35],[-75,38],[-70,42],[-65,44],[-61,46],
    // Maritime Canada & Labrador
    [-66,47],[-70,52],[-64,54],[-68,62],
    // Arctic coast (simplified)
    [-72,66],[-80,70],[-100,75],[-120,75],[-140,72],[-158,70],[-168,63],
  ]]},

  // ── ALASKA (separate polygon so Gulf of Alaska coast is right) ─────────────
  { id: 'ak', color: '#5577ee', glow: '#3355bb', polys: [[
    [-168,63],[-158,58],[-152,58],[-148,60],[-144,60],
    [-136,58],[-132,56],[-141,68],[-153,72],[-164,70],[-168,63],
  ]]},

  // ── GREENLAND ──────────────────────────────────────────────────────────────
  { id: 'gl', color: '#99bbdd', glow: '#7799bb', polys: [[
    [-50,61],[-43,65],[-24,70],[-18,76],[-20,80],
    [-42,84],[-62,83],[-72,78],[-74,70],[-68,63],[-58,61],[-50,61],
  ]]},

  // ── SOUTH AMERICA ──────────────────────────────────────────────────────────
  { id: 'sa', color: '#00bbaa', glow: '#009988', polys: [[
    // CW from NE Venezuela coast
    [-73,12],[-60,12],[-52,6],[-48,2],
    // E coast going south (Amazon bulge, then Recife, Rio, etc.)
    [-35,-4],[-35,-8],[-38,-14],[-40,-20],[-43,-23],
    [-50,-28],[-52,-34],[-56,-38],[-60,-42],[-62,-52],
    // Cape Horn area
    [-68,-55],
    // W coast going north (Chile, Peru, Ecuador)
    [-72,-50],[-74,-44],[-72,-36],[-70,-30],[-70,-18],[-78,-8],[-80,0],
    // Colombia Pacific coast → Caribbean
    [-78,8],[-76,10],[-73,12],
  ]]},

  // ── EUROPE ─────────────────────────────────────────────────────────────────
  // Main European mass (W Europe including Iberian, Italian, Greek peninsulas)
  { id: 'eu', color: '#6699dd', glow: '#4477bb', polys: [
    // Main mass: Portugal → Mediterranean → Aegean → Turkey coast → E Europe → Arctic → back
    [[
      [-10,36],[-8,38],[-9,44],[-4,44],[0,44],
      // France/Mediterranean coast
      [3,43],[5,43],[8,44],[12,44],
      // Italy boot
      [14,40],[16,38],[18,40],
      // Adriatic/Balkans/Greece
      [20,40],[22,37],[24,38],
      // Thrace / W Turkey (Europe side)
      [26,40],[28,42],[30,44],
      // Romania/Ukraine Black Sea
      [32,46],[34,46],[36,46],
      // Russia south (Ukraine/Russia)
      [40,48],[46,50],[50,52],[60,58],
      // Urals boundary → Russia north coast (simplified)
      [65,68],[60,70],[50,70],[40,70],[30,70],
      // N Norway/Scandinavia top
      [28,72],[20,70],[18,70],
      // Scandinavia south (simplified — Baltic inside counts as land)
      [14,56],[10,58],[8,56],[5,56],[4,52],
      // W Europe Atlantic coast north to south
      [2,52],[0,50],[-2,48],[-4,46],[-5,44],[-4,44],
      // Closing back to Portugal
      [-9,44],[-8,38],[-10,36],
    ]],
    // Iceland (separate island)
    [[-24,63],[-13,63],[-13,66],[-22,66],[-24,63]],
  ]},

  // ── AFRICA ─────────────────────────────────────────────────────────────────
  { id: 'af', color: '#44bb77', glow: '#229955', polys: [[
    // CW from Morocco NW corner
    [-6,36],[0,36],[10,37],[12,32],
    // Libya → Egypt N coast
    [25,32],[32,30],
    // Red Sea coast of Africa (going south)
    [32,22],[36,16],[40,12],[42,12],
    // Djibouti → Horn of Africa (Somalia)
    [44,12],[50,12],[52,10],
    // East Africa coast going south
    [44,2],[40,-10],[38,-20],[36,-26],
    [32,-30],[28,-35],
    // Cape of Good Hope area
    [20,-35],[18,-34],[16,-30],
    // West Africa coast going north
    [12,-18],[12,-6],[10,2],[8,4],[4,4],
    [2,4],[-2,4],[-4,4],[-8,4],[-12,6],
    // Senegal/Mauritania/Morocco W coast
    [-16,12],[-17,16],[-16,20],[-14,24],[-8,28],[-6,32],[-6,36],
  ]]},

  // ── MADAGASCAR ─────────────────────────────────────────────────────────────
  { id: 'mg', color: '#44bb77', glow: '#229955', polys: [[
    [44,-12],[50,-14],[50,-20],[44,-25],[44,-18],[44,-12],
  ]]},

  // ── ASIA — MAIN MASS ───────────────────────────────────────────────────────
  // The main Eurasian mass E of ~26°E longitude (Turkey, Middle East, Russia, China)
  { id: 'as', color: '#4488cc', glow: '#226699', polys: [[
    // Start at Bosphorus/W Turkey, go CW along S coast of Asia
    [26,40],  // W Turkey / Aegean coast
    [36,36],  // Turkey S coast (Mediterranean)
    [36,28],  // Israel/Lebanon coast (going south)
    // Sinai / Arabia start:
    [34,28],[34,22],[36,16],[44,12],[50,12],
    // Gulf of Aden to India approach — jump over open water with broad stroke
    [60,22],[60,14],[58,8],[76,8],
    // India S tip → E coast
    [80,8],[80,14],[80,20],[86,22],[88,24],
    // Bangladesh → Burma/Myanmar coast
    [92,22],[98,16],
    // Malacca / W Malaysia / Thai coast
    [100,6],[104,2],[104,6],[108,10],
    // Vietnam/Cambodia S coast → N Vietnam
    [104,10],[108,12],[108,14],[108,18],
    // China S coast
    [110,20],[114,22],[120,26],[122,30],
    // Shanghai → Yellow Sea
    [122,36],[122,38],
    // Korea S tip → E coast of Korea
    [128,36],[130,40],
    // Russia far east (Vladivostok → Bering Strait)
    [132,46],[140,48],[142,50],[146,52],[148,56],
    [140,60],[140,68],[150,72],[160,68],[168,66],
    // Now go WEST along Arctic coast (top of Russia, simplified)
    [170,72],[150,74],[130,74],[110,74],[100,75],
    [80,72],[70,70],[60,68],[50,70],[40,70],
    [36,46],[34,46],[32,46],
    // Black Sea north coast → Turkey north coast
    [30,44],[28,42],
    // Close back to W Turkey
    [26,40],
  ]]},

  // ── ARABIAN PENINSULA ──────────────────────────────────────────────────────
  { id: 'ar', color: '#8866bb', glow: '#664499', polys: [[
    [36,28],[44,12],[50,12],[60,22],[60,26],
    [56,26],[54,24],[50,26],[48,28],
    // Persian Gulf / Iraq coast
    [44,30],[38,30],[36,28],
  ]]},

  // ── INDIAN SUBCONTINENT ────────────────────────────────────────────────────
  // Already included in ASIA polygon above, but add a brighter overlay for India
  { id: 'in', color: '#55aadd', glow: '#3388bb', polys: [[
    [66,24],[68,24],[72,22],[76,8],[80,8],[80,20],
    [86,22],[88,24],[92,22],[84,28],[80,28],[76,28],
    [72,28],[68,28],[66,24],
  ]]},

  // ── JAPAN ──────────────────────────────────────────────────────────────────
  { id: 'jp', color: '#5599ee', glow: '#3377cc', polys: [
    // Honshu
    [[130,32],[132,34],[136,36],[138,36],[140,38],[142,40],[140,42],[136,40],[132,36],[130,32]],
    // Hokkaido
    [[140,42],[144,44],[146,44],[144,44],[140,44],[138,42],[140,42]],
    // Kyushu
    [[130,32],[132,32],[132,34],[130,34],[130,32]],
  ]},

  // ── GREAT BRITAIN + IRELAND ────────────────────────────────────────────────
  { id: 'uk', color: '#5599dd', glow: '#3377bb', polys: [
    [[-6,50],[-2,50],[-2,54],[-4,58],[-2,58],[0,58],[-2,60],[-6,58],[-8,54],[-6,50]],
    [[-10,52],[-6,52],[-6,54],[-10,54],[-10,52]],
  ]},

  // ── AUSTRALIA ──────────────────────────────────────────────────────────────
  { id: 'au', color: '#bb8844', glow: '#996622', polys: [[
    // CW from SW corner
    [114,-34],[118,-36],[122,-34],[126,-34],[130,-32],
    [134,-32],[136,-36],[138,-36],[140,-36],
    // SE coast
    [148,-38],[150,-38],[152,-30],[152,-24],
    // Queensland / N Australia
    [148,-20],[142,-10],[136,-12],[132,-12],
    // NT / Kimberley
    [128,-14],[124,-16],[118,-20],[116,-32],[114,-34],
  ]]},

  // ── NEW ZEALAND ────────────────────────────────────────────────────────────
  { id: 'nz', color: '#55aacc', glow: '#3388aa', polys: [
    [[166,-46],[170,-46],[172,-44],[172,-42],[170,-38],[168,-36],[166,-38],[166,-46]],
    [[172,-44],[174,-42],[174,-38],[172,-40],[172,-44]],
  ]},

  // ── SRI LANKA ──────────────────────────────────────────────────────────────
  { id: 'sl', color: '#55aadd', glow: '#3388bb', polys: [
    [[80,6],[82,6],[82,8],[80,10],[80,6]],
  ]},

  // ── TAIWAN ─────────────────────────────────────────────────────────────────
  { id: 'tw', color: '#5599ee', glow: '#3377cc', polys: [
    [[120,22],[122,22],[122,24],[120,24],[120,22]],
  ]},

  // ── BORNEO (Kalimantan) ────────────────────────────────────────────────────
  { id: 'bo', color: '#44aacc', glow: '#228899', polys: [[
    [108,2],[116,2],[118,4],[118,6],[116,8],[112,6],[108,4],[108,2],
  ]]},

  // ── SUMATRA ────────────────────────────────────────────────────────────────
  { id: 'su', color: '#44aacc', glow: '#228899', polys: [[
    [96,4],[98,4],[100,2],[104,0],[108,2],[106,4],[102,6],[98,6],[96,4],
  ]]},

  // ── JAVA ───────────────────────────────────────────────────────────────────
  { id: 'jv', color: '#44aacc', glow: '#228899', polys: [[
    [106,-6],[108,-6],[110,-8],[112,-8],[114,-8],[112,-6],[108,-6],[106,-6],
  ]]},

  // ── PHILIPPINES (simplified) ───────────────────────────────────────────────
  { id: 'ph', color: '#5599ee', glow: '#3377cc', polys: [
    [[120,10],[122,10],[122,14],[120,14],[120,10]],
    [[124,8],[126,8],[126,12],[124,12],[124,8]],
  ]},

  // ── ANTARCTICA ─────────────────────────────────────────────────────────────
  { id: 'an', color: '#aaccee', glow: '#88aacc', polys: [[
    [-180,-68],[180,-68],[180,-90],[-180,-90],[-180,-68],
  ]]},
]

// Pre-process: flatten each polygon to [[lon,lat], ...] format
const REGIONS_PROCESSED = LAND_REGIONS.map(r => ({
  ...r,
  polys: r.polys.map(poly =>
    // Each poly is either [[lon,lat],...] already, or needs flattening
    Array.isArray(poly[0]) ? poly : poly
  ),
}))

function getLandRegion(lon, lat) {
  for (const region of REGIONS_PROCESSED) {
    for (const poly of region.polys) {
      if (pip(lon, lat, poly)) return region
    }
  }
  return null
}

// ─── Shape visual weight compensation ────────────────────────────────────────
const SHAPE_COMP = { circle: 1.0, triangle: 1.35, square: 1.15, diamond: 1.25 }
const SHAPES = ['circle', 'triangle', 'square', 'diamond']
const PALETTE = ['#4488cc', '#9966ee', '#00aa99', '#5577ee', '#aaaacc']

// ─── Component ────────────────────────────────────────────────────────────────
export default function ParticleCosmos({ state = 'drift', speedMultiplier = 1.0 }) {
  const canvasRef     = useRef(null)
  const requestRef    = useRef(null)
  const particlesRef  = useRef([])
  const prevStateRef  = useRef(null)
  const mouseRef      = useRef({ x: null, y: null, radius: 120 })
  const globeAngleRef = useRef(0.3)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    let width  = (canvas.width  = canvas.offsetWidth)
    let height = (canvas.height = canvas.offsetHeight)

    if (prevStateRef.current !== state) {
      particlesRef.current = []
      prevStateRef.current = state
    }

    if (particlesRef.current.length === 0) {
      if (state === 'globe') {
        const N = 3000
        const goldenAngle = Math.PI * (3 - Math.sqrt(5))
        const pts = []

        for (let i = 0; i < N; i++) {
          const phiG   = Math.acos(1 - (2 * (i + 0.5)) / N)
          const thetaG = goldenAngle * i

          const lat     = 90 - (phiG * 180) / Math.PI
          const lon     = ((thetaG * 180) / Math.PI) % 360
          const lonNorm = lon > 180 ? lon - 360 : lon

          const region  = getLandRegion(lonNorm, lat)
          const isLand  = region !== null

          const sx = Math.sin(phiG) * Math.cos(thetaG)
          const sy = Math.cos(phiG)
          const sz = Math.sin(phiG) * Math.sin(thetaG)

          // Keep original shape distribution
          const shape = isLand
            ? (Math.random() < 0.6 ? 'circle' : SHAPES[Math.floor(Math.random() * SHAPES.length)])
            : 'circle'

          pts.push({
            sx, sy, sz,
            color:     isLand ? region.color  : '#0a1e38',
            glowColor: isLand ? region.glow   : '#0d2a50',
            comp:      SHAPE_COMP[shape] || 1.0,
            // Land: 1.6–3.8px; ocean: 0.5–1.4px
            size:  isLand ? (Math.random() * 2.2 + 1.6) : (Math.random() * 0.9 + 0.5),
            shape,
            isLand,
          })
        }
        particlesRef.current = pts

      } else {
        const N = 900
        const pts = []
        for (let i = 0; i < N; i++) {
          pts.push({
            x: Math.random() * width, y: Math.random() * height,
            vx: (Math.random() - 0.5) * 0.8, vy: (Math.random() - 0.5) * 0.8,
            size: Math.random() * 3.2 + 1.2,
            color: PALETTE[Math.floor(Math.random() * PALETTE.length)],
            shape: SHAPES[Math.floor(Math.random() * SHAPES.length)],
            targetX: null, targetY: null,
            angle: Math.random() * Math.PI * 2,
            phi: Math.random() * Math.PI,
            radius: Math.random() * Math.min(width, height) * 0.4,
            speed: (Math.random() * 0.02 + 0.005) * speedMultiplier,
          })
        }
        particlesRef.current = pts
      }
    }

    const setMorphTargets = (s) => {
      const cx = width / 2, cy = height / 2
      const pts = particlesRef.current
      pts.forEach((p, i) => {
        if (s === 'dandelion') {
          const quota = Math.floor(pts.length * 0.2)
          if (i < quota) {
            const a = Math.random() * Math.PI * 2, r = Math.random() * 24
            p.targetX = cx + Math.cos(a) * r; p.targetY = cy + Math.sin(a) * r
          } else {
            const stalk = (i - quota) % 40
            const a = (stalk / 40) * Math.PI * 2 + (Math.random() - 0.5) * 0.05
            const len = Math.pow(Math.random(), 0.7) * 110 + 20
            p.targetX = cx + Math.cos(a) * len; p.targetY = cy + Math.sin(a) * len
          }
        } else { p.targetX = null; p.targetY = null }
      })
    }

    if (state !== 'globe') setMorphTargets(state)

    const handleResize = () => {
      width = canvas.width = canvas.offsetWidth
      height = canvas.height = canvas.offsetHeight
      if (state !== 'globe') setMorphTargets(state)
    }
    window.addEventListener('resize', handleResize)

    // ── Draw crisp shape ──────────────────────────────────────────────────────
    const drawShape = (p, x, y, sz) => {
      ctx.beginPath()
      if (p.shape === 'circle') {
        ctx.arc(x, y, sz / 2, 0, Math.PI * 2)
      } else if (p.shape === 'triangle') {
        const r = sz * 0.60
        ctx.moveTo(x, y - r)
        ctx.lineTo(x + r * 0.866, y + r * 0.5)
        ctx.lineTo(x - r * 0.866, y + r * 0.5)
        ctx.closePath()
      } else if (p.shape === 'square') {
        const h = sz * 0.44
        ctx.rect(x - h, y - h, h * 2, h * 2)
      } else {
        const h = sz * 0.56
        ctx.moveTo(x, y - h); ctx.lineTo(x + h, y)
        ctx.lineTo(x, y + h); ctx.lineTo(x - h, y)
        ctx.closePath()
      }
      ctx.fill()
    }

    // ── Draw soft glow halo ───────────────────────────────────────────────────
    const drawGlow = (x, y, sz, glowColor, alpha) => {
      // Reduced halo size: 2.0× instead of 3.2×
      const r = sz * 2.0
      const g = ctx.createRadialGradient(x, y, 0, x, y, r)
      const a1 = Math.round(alpha * 255).toString(16).padStart(2, '0')
      const a2 = Math.round(alpha * 0.15 * 255).toString(16).padStart(2, '0')
      g.addColorStop(0,   glowColor + a1)
      g.addColorStop(0.5, glowColor + a2)
      g.addColorStop(1,   glowColor + '00')
      ctx.fillStyle = g
      ctx.beginPath()
      ctx.arc(x, y, r, 0, Math.PI * 2)
      ctx.fill()
    }

    // ── Animation ─────────────────────────────────────────────────────────────
    const animate = () => {
      ctx.clearRect(0, 0, width, height)
      const mouse = mouseRef.current
      const cx = width / 2, cy = height / 2

      if (state === 'globe') {
        const globeR = Math.min(width, height) * 0.46
        const PERSP  = 2.8

        if (!mouse.x) globeAngleRef.current += 0.0022
        const cosA = Math.cos(globeAngleRef.current)
        const sinA = Math.sin(globeAngleRef.current)

        // Project
        const projected = particlesRef.current.map(p => {
          const rx = p.sx * cosA - p.sz * sinA
          const ry = p.sy
          const rz = p.sx * sinA + p.sz * cosA
          const scale = PERSP / (PERSP - rz * 0.55)
          return {
            p,
            px: cx + rx * globeR * scale,
            py: cy - ry * globeR * scale,   // negate Y: canvas grows downward, north must be up
            rz,
            depthT: (rz + 1) / 2,
          }
        }).sort((a, b) => a.rz - b.rz)

        // ── Glow pass ─────────────────────────────────────────────────────────
        // Reduced brightness: ~25% dimmer vs previous
        projected.forEach(({ p, px, py, rz, depthT }) => {
          if (rz < -0.12) return

          const minOp = p.isLand ? 0.18 : 0.08
          const maxOp = p.isLand ? 0.45 : 0.22
          const opacity = Math.min(0.95, (minOp + depthT * (maxOp - minOp)) * p.comp)

          let x = px, y = py
          if (mouse.x !== null) {
            const dx = px - mouse.x, dy = py - mouse.y
            const d = Math.sqrt(dx*dx + dy*dy)
            if (d < 80) { const f = (80 - d) / 80; x += dx/d*f*5; y += dy/d*f*5 }
          }

          const sz = p.size * (0.68 + depthT * 0.32)
          // Glow is 45% of particle opacity (was 70%)
          drawGlow(x, y, sz, p.glowColor, opacity * 0.42)
        })

        // ── Shape pass ────────────────────────────────────────────────────────
        projected.forEach(({ p, px, py, rz, depthT }) => {
          if (rz < -0.12) return

          const minOp = p.isLand ? 0.18 : 0.08
          const maxOp = p.isLand ? 0.45 : 0.22
          const opacity = Math.min(0.95, (minOp + depthT * (maxOp - minOp)) * p.comp)

          let x = px, y = py
          if (mouse.x !== null) {
            const dx = px - mouse.x, dy = py - mouse.y
            const d = Math.sqrt(dx*dx + dy*dy)
            if (d < 80) { const f = (80-d)/80; x += dx/d*f*5; y += dy/d*f*5 }
          }

          const sz = p.size * (0.68 + depthT * 0.32)
          const hex = Math.round(opacity * 255).toString(16).padStart(2, '0')
          ctx.fillStyle = p.color + hex
          drawShape(p, x, y, sz)
        })

      } else {
        particlesRef.current.forEach(p => {
          if (state === 'vortex') {
            p.angle += p.speed * 2
            const tR = p.radius * 0.8
            p.x += (cx + Math.cos(p.angle) * tR - p.x) * 0.08
            p.y += (cy + Math.sin(p.angle) * tR - p.y) * 0.08
          } else if (p.targetX !== null) {
            const dx = p.targetX - p.x, dy = p.targetY - p.y
            p.x += dx * 0.06; p.y += dy * 0.06
            const d = Math.sqrt(dx*dx+dy*dy)
            if (d < 10) { p.x += (Math.random()-.5)*.25; p.y += (Math.random()-.5)*.25 }
          } else {
            p.x += p.vx * speedMultiplier; p.y += p.vy * speedMultiplier
            if (p.x < 0 || p.x > width) p.vx *= -1
            if (p.y < 0 || p.y > height) p.vy *= -1
          }
          if (mouse.x !== null) {
            const dx = p.x - mouse.x, dy = p.y - mouse.y
            const d = Math.sqrt(dx*dx+dy*dy)
            if (d < mouse.radius) {
              const f = (mouse.radius - d) / mouse.radius
              const a = Math.atan2(dy, dx)
              p.x += Math.cos(a)*f*4; p.y += Math.sin(a)*f*4
            }
          }
          ctx.fillStyle = p.color
          drawShape(p, p.x, p.y, p.size)
        })
      }

      requestRef.current = requestAnimationFrame(animate)
    }

    animate()

    const onMove = (e) => {
      const r = canvas.getBoundingClientRect()
      mouseRef.current.x = e.clientX - r.left
      mouseRef.current.y = e.clientY - r.top
    }
    const onLeave = () => { mouseRef.current.x = null; mouseRef.current.y = null }
    canvas.addEventListener('mousemove', onMove)
    canvas.addEventListener('mouseleave', onLeave)

    return () => {
      window.removeEventListener('resize', handleResize)
      if (requestRef.current) cancelAnimationFrame(requestRef.current)
      canvas.removeEventListener('mousemove', onMove)
      canvas.removeEventListener('mouseleave', onLeave)
    }
  }, [state, speedMultiplier])

  return (
    <canvas
      ref={canvasRef}
      style={{
        width: '100%', height: '100%',
        display: 'block',
        position: 'absolute', top: 0, left: 0,
        pointerEvents: 'auto', zIndex: 0,
      }}
    />
  )
}
