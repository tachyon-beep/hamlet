<template>
  <div class="curriculum-tracker">
    <h4>Curriculum Progress</h4>
    <div class="stage-display">
      <span class="stage-label">Stage {{ currentStage }}/5</span>
      <span class="stage-description">{{ stageDescription }}</span>
    </div>
    <div class="progress-bar">
      <div class="progress-fill" :style="{ width: progressPercent + '%' }"></div>
    </div>
    <div class="progress-text">{{ stepsAtStage }} / {{ minStepsRequired }} steps</div>
  </div>
</template>

<script>
export default {
  name: 'CurriculumTracker',
  props: {
    currentStage: {
      type: Number,
      default: 1,
    },
    stepsAtStage: {
      type: Number,
      default: 0,
    },
    minStepsRequired: {
      type: Number,
      default: 1000,
    },
  },
  computed: {
    stageDescription() {
      const descriptions = {
        1: 'Basic Needs (Energy, Hygiene)',
        2: 'Add Hunger Management',
        3: 'Add Economic Planning',
        4: 'Full Complexity (All Meters)',
        5: 'SPARSE REWARDS - Graduation!',
      }
      return descriptions[this.currentStage] || 'Unknown Stage'
    },
    progressPercent() {
      return Math.min((this.stepsAtStage / this.minStepsRequired) * 100, 100)
    },
  },
}
</script>

<style scoped>
.curriculum-tracker {
  margin: 10px 0;
  padding: 10px;
  border: 1px solid #ccc;
  border-radius: 4px;
}

.stage-display {
  display: flex;
  justify-content: space-between;
  margin-bottom: 8px;
}

.stage-label {
  font-weight: bold;
  font-size: 16px;
}

.progress-bar {
  height: 20px;
  background: #e0e0e0;
  border-radius: 4px;
  overflow: hidden;
  margin-bottom: 4px;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #3b82f6, #10b981);
  transition: width 0.3s ease;
}

.progress-text {
  font-size: 12px;
  color: #666;
  text-align: right;
}
</style>
