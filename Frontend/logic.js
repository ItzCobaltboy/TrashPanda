
// logic.js

export function generateTrashcanCSV({
  cityMap,
  trashcanCount = 20,
  fillDays = 200,
  fillRateRange = [3, 20],
  minEmptyThreshold = 80,
  noiseStdFrac = 0.2,
  seasonalMeanNoiseFrac = 0.1,
  startDateStr = "2023-01-01"
}) {
  const START_DATE = new Date(startDateStr);
  const trashcanIDs = Array.from({ length: trashcanCount }, (_, i) => `trashcan_${i + 1}`);
  const edgeList = cityMap.edges.map(e => e.id);
  const records = [];
  const TRASHCAN_FILL_RATES = {};
  const EDGE_TO_TRASHCANS = {};
  const FILL_LEVEL_TRACKER = {}; // current fill per trashcan

  for (const id of trashcanIDs) {
    const edgeID = edgeList[Math.floor(Math.random() * edgeList.length)];
    const baseMeanFill = rand(...fillRateRange);
    TRASHCAN_FILL_RATES[id] = baseMeanFill;

    if (!EDGE_TO_TRASHCANS[edgeID]) EDGE_TO_TRASHCANS[edgeID] = [];
    EDGE_TO_TRASHCANS[edgeID].push(id);

    let currentFill = rand(0, 30);
    FILL_LEVEL_TRACKER[id] = currentFill;
    const dayValues = [];

    for (let i = 0; i < fillDays; i++) {
      const date = new Date(START_DATE);
      date.setDate(date.getDate() + i);
      const seasonalMultiplier = date.getDay() >= 5 ? 1.2 : 1.0;
      const seasonal = baseMeanFill * seasonalMultiplier;
      const noise = randNormal(0, seasonal * seasonalMeanNoiseFrac);
      const seasonalMean = Math.max(0, Math.min(seasonal + noise, 100));
      const dailyAdd = Math.max(0, Math.min(randNormal(seasonalMean, seasonalMean * noiseStdFrac), 100));

      currentFill = Math.min(currentFill + dailyAdd, 100);
      if (currentFill >= minEmptyThreshold && Math.random() < 0.3) {
        currentFill = 0;
      }

      dayValues.push(currentFill.toFixed(2));
      FILL_LEVEL_TRACKER[id] = currentFill;
    }

    records.push([id, edgeID, ...dayValues]);
  }

  const header = ["trashcanID", "edgeID"];
  for (let i = 0; i < fillDays; i++) {
    const d = new Date(START_DATE);
    d.setDate(d.getDate() + i);
    header.push(d.toISOString().split("T")[0]);
  }

  return {
    csv: [header.join(","), ...records.map(row => row.join(","))].join("\n"),
    edgeToTrashcans: EDGE_TO_TRASHCANS,
    currentFill: FILL_LEVEL_TRACKER,
    baseRates: TRASHCAN_FILL_RATES
  };
}

export function generateNextFill(prevFill, baseMean = rand(3, 20), noiseStdFrac = 0.2, seasonalMeanNoiseFrac = 0.1) {
  const today = new Date();
  const seasonalMultiplier = today.getDay() >= 5 ? 1.2 : 1.0;
  const seasonal = baseMean * seasonalMultiplier;
  const noise = randNormal(0, seasonal * seasonalMeanNoiseFrac);
  const seasonalMean = Math.max(0, Math.min(seasonal + noise, 100));
  const dailyAdd = Math.max(0, Math.min(randNormal(seasonalMean, seasonalMean * noiseStdFrac), 100));
  return Math.min(prevFill + dailyAdd, 100);
}

function rand(min, max) {
  return Math.random() * (max - min) + min;
}

function randNormal(mean, std) {
  let u = 1 - Math.random();
  let v = 1 - Math.random();
  return mean + std * Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}
