/* ── KYCGuard landing page logic ─────────────────────────── */

(() => {
  "use strict";

  // Scroll indicator → smooth-scroll down to the "How It Works" section
  const indicator = document.getElementById("scroll-indicator");
  const howSection = document.getElementById("how-it-works");
  if (indicator && howSection) {
    const scrollToHow = () => howSection.scrollIntoView({ behavior: "smooth" });
    indicator.addEventListener("click", scrollToHow);
    indicator.addEventListener("keydown", e => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        scrollToHow();
      }
    });
  }

  // ── Kinetic heading: staggered per-letter entrance ─────────
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const heroTitle = document.getElementById("hero-title");
  if (heroTitle && !reduceMotion) {
    let delay = 0.15;
    heroTitle.querySelectorAll(".logo-fin, .logo-shield").forEach(word => {
      const text = word.textContent;
      word.setAttribute("aria-label", word.textContent);
      word.textContent = "";
      [...text].forEach(ch => {
        const s = document.createElement("span");
        s.className = "letter";
        s.textContent = ch;
        s.style.animationDelay = delay.toFixed(2) + "s";
        word.appendChild(s);
        delay += 0.05;
      });
    });
  }

  // ── One-time typewriter reveal for the hero paragraph ─────
  // Runs once on load (never on scroll). Tags & whitespace are inserted
  // instantly; visible characters are typed one at a time.
  const subtitle = document.querySelector(".hero-subtitle");
  if (subtitle && !reduceMotion) {
    const tokens = subtitle.innerHTML.match(/<[^>]*>|[^<>]+/g) || [subtitle.innerHTML];
    const items = [];
    tokens.forEach(tok => {
      if (tok[0] === "<" || /^\s+$/.test(tok)) {
        items.push(tok); // tags & whitespace: instant
      } else {
        for (const ch of tok) items.push(ch);
      }
    });

    subtitle.innerHTML = "";

    const cursor = document.createElement("span");
    cursor.className = "type-cursor";
    cursor.setAttribute("aria-hidden", "true");
    subtitle.appendChild(cursor);

    const SPEED = 28;
    let i = 0;
    (function type() {
      if (i >= items.length) {
        cursor.remove();
        return;
      }
      const piece = items[i++];
      subtitle.insertAdjacentHTML("beforeend", piece);
      if (piece[0] === "<" || /^\s+$/.test(piece)) {
        type();
      } else {
        setTimeout(type, SPEED);
      }
    })();
  }

  // ── Magnetic "Try Demo" button ─────────────────────────────
  const cta = document.querySelector(".hero-cta");
  if (cta && window.matchMedia("(pointer: fine)").matches) {
    const strength = 8;
    cta.addEventListener("mousemove", e => {
      const r = cta.getBoundingClientRect();
      const dx = e.clientX - (r.left + r.width / 2);
      const dy = e.clientY - (r.top + r.height / 2);
      const tx = Math.max(-strength, Math.min(strength, dx * 0.25));
      const ty = Math.max(-strength, Math.min(strength, dy * 0.25));
      cta.style.transform = `translate(${tx}px, ${ty}px)`;
    });
    cta.addEventListener("mouseleave", () => {
      cta.style.transform = "";
    });
  }

  // ── Cursor glow that follows the pointer with easing ───────
  const hero = document.querySelector(".hero");
  if (hero && window.matchMedia("(pointer: fine)").matches) {
    const glow = document.createElement("div");
    glow.className = "cursor-glow";
    glow.setAttribute("aria-hidden", "true");
    hero.appendChild(glow);
    let cx = window.innerWidth / 2;
    let cy = window.innerHeight / 2;
    let tx = cx;
    let ty = cy;
    hero.addEventListener("mousemove", e => {
      tx = e.clientX;
      ty = e.clientY;
      hero.classList.add("cursor-active");
    });
    (function follow() {
      cx += (tx - cx) * 0.14;
      cy += (ty - cy) * 0.14;
      glow.style.transform = `translate(${cx.toFixed(1)}px, ${cy.toFixed(1)}px)`;
      requestAnimationFrame(follow);
    })();
  }

  // ── Intersection Observer for scroll-triggered reveals ────
  const observerOptions = { threshold: 0.15, rootMargin: "0px 0px -40px 0px" };
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add("visible");
        observer.unobserve(entry.target);
      }
    });
  }, observerOptions);

  document.querySelectorAll(".reveal").forEach(el => observer.observe(el));

  // Render Lucide icons
  if (window.lucide) lucide.createIcons();
})();