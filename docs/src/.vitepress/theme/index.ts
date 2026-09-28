// .vitepress/theme/index.ts
import { h } from 'vue'
import DefaultTheme from 'vitepress/theme'
import type { Theme as ThemeConfig } from 'vitepress'
import 'virtual:mathjax-styles.css';

import { 
  NolebaseEnhancedReadabilitiesMenu, 
  NolebaseEnhancedReadabilitiesScreenMenu, 
} from '@nolebase/vitepress-plugin-enhanced-readabilities/client'

import VersionPicker from "@/VersionPicker.vue"
import AuthorBadge from '@/AuthorBadge.vue'
import Authors from '@/Authors.vue'
import Banner from '@/Banner.vue'

// Snapshot of HTMXObjects/assets/vitepress/htmxo-embed.ts, synced from
// HTMXObjects@devibe c9a1bdcd1c5933fa936c409646278ca0c3bf3f18 (see the
// note in docs/make.jl). Don't edit in place — edit the upstream and
// re-sync the file manually.
import { setupHtmxoEmbed } from './htmxo-embed'

import { enhanceAppWithTabs } from 'vitepress-plugin-tabs/client'

import '@nolebase/vitepress-plugin-enhanced-readabilities/client/style.css'
import './style.css' // You could setup your own, or else a default will be copied.
import './docstrings.css' // You could setup your own, or else a default will be copied.

export const Theme: ThemeConfig = {
  extends: DefaultTheme,
  Layout() {
    return h(DefaultTheme.Layout, null, {
      'layout-bottom': () => h(Banner),
      'nav-bar-content-after': () => [
        h(NolebaseEnhancedReadabilitiesMenu), // Enhanced Readabilities menu
      ],
      // A enhanced readabilities menu for narrower screens (usually smaller than iPad Mini)
      'nav-screen-content-after': () => h(NolebaseEnhancedReadabilitiesScreenMenu),
    })
  },
  enhanceApp({ app, router, siteData }) {
    enhanceAppWithTabs(app);
    app.component('VersionPicker', VersionPicker);
    app.component('AuthorBadge', AuthorBadge)
    app.component('Authors', Authors)
    // HTMXObjects embed wiring: data-hx-base resolution + SPA route
    // re-process + .htmxo-embed link rewriting. Defaults the proxy
    // prefix to `/live-reactiveobjects` (matches the Vite proxy in
    // config.mts and the committed recordings under
    // public/live-reactiveobjects/).
    setupHtmxoEmbed(router, { proxyPrefix: '/live-reactiveobjects' });
  }
}
export default Theme