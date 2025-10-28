import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'
import './assets/styles.css'
import './styles/variables.css'  // Design tokens as CSS custom properties

const app = createApp(App)
const pinia = createPinia()

app.use(pinia)
app.mount('#app')
