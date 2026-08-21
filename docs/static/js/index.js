document.addEventListener('DOMContentLoaded', () => {
  const burger = document.querySelector('.navbar-burger');
  const menu = document.querySelector('.navbar-menu');

  if (burger && menu) {
    burger.addEventListener('click', () => {
      const isOpen = burger.classList.toggle('is-active');
      menu.classList.toggle('is-active', isOpen);
      burger.setAttribute('aria-expanded', String(isOpen));
    });

    menu.querySelectorAll('a').forEach((link) => {
      link.addEventListener('click', () => {
        burger.classList.remove('is-active');
        menu.classList.remove('is-active');
        burger.setAttribute('aria-expanded', 'false');
      });
    });
  }

  const story = document.querySelector('[data-story-carousel]');
  if (story) {
    const panels = [...story.querySelectorAll('[data-story-panel]')];
    const selectors = [...story.querySelectorAll('[data-story-index]')];
    let currentIndex = 0;
    let storyIsVisible = false;

    const pauseStoryVideos = () => {
      story.querySelectorAll('[data-story-video]').forEach((video) => video.pause());
    };

    const playCurrentStory = (restart = true) => {
      const panel = panels[currentIndex];
      if (!panel || !storyIsVisible) return;
      panel.querySelectorAll('[data-story-video]').forEach((video) => {
        if (restart) video.currentTime = 0;
        video.play().catch(() => {});
      });
    };

    const showStory = (nextIndex, restart = true) => {
      currentIndex = (nextIndex + panels.length) % panels.length;
      pauseStoryVideos();

      panels.forEach((panel, index) => {
        const active = index === currentIndex;
        panel.hidden = !active;
        panel.classList.toggle('is-active', active);
      });

      selectors.forEach((selector) => {
        const active = Number(selector.dataset.storyIndex) === currentIndex;
        selector.classList.toggle('is-active', active);
        if (selector.getAttribute('role') === 'tab') selector.setAttribute('aria-selected', String(active));
      });

      playCurrentStory(restart);
    };

    selectors.forEach((selector) => selector.addEventListener('click', () => showStory(Number(selector.dataset.storyIndex))));
    story.querySelector('[data-story-prev]')?.addEventListener('click', () => showStory(currentIndex - 1));
    story.querySelector('[data-story-next]')?.addEventListener('click', () => showStory(currentIndex + 1));
    story.addEventListener('keydown', (event) => {
      if (event.key === 'ArrowLeft') showStory(currentIndex - 1);
      if (event.key === 'ArrowRight') showStory(currentIndex + 1);
    });

    let touchStartX = 0;
    story.addEventListener('touchstart', (event) => { touchStartX = event.changedTouches[0].clientX; }, { passive: true });
    story.addEventListener('touchend', (event) => {
      const distance = event.changedTouches[0].clientX - touchStartX;
      if (Math.abs(distance) > 55) showStory(currentIndex + (distance < 0 ? 1 : -1));
    }, { passive: true });

    const storyObserver = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        storyIsVisible = entry.isIntersecting;
        if (storyIsVisible) playCurrentStory(true);
        else pauseStoryVideos();
      });
    }, { threshold: 0.35 });
    storyObserver.observe(story);
    showStory(0, false);
  }

  const domainRows = [...document.querySelectorAll('[data-domain-row]')];
  const stopDomain = (row, reset = true) => {
    row.classList.remove('is-playing');
    row.querySelectorAll('video').forEach((video) => {
      video.pause();
      if (reset) video.currentTime = 0;
    });
  };
  const stopAllDomains = (except = null) => domainRows.forEach((row) => {
    if (row !== except) stopDomain(row);
  });
  const playDomain = (row) => {
    stopAllDomains(row);
    row.classList.add('is-playing');
    row.querySelectorAll('video').forEach((video) => {
      video.currentTime = 0;
      video.play().catch(() => {});
    });
  };

  domainRows.forEach((row) => {
    row.addEventListener('mouseenter', () => playDomain(row));
    row.addEventListener('mouseleave', () => stopDomain(row));
    row.addEventListener('focusin', () => playDomain(row));
    row.addEventListener('focusout', (event) => {
      if (!row.contains(event.relatedTarget)) stopDomain(row);
    });
    row.addEventListener('click', (event) => {
      event.stopPropagation();
      playDomain(row);
    });
    row.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        row.classList.contains('is-playing') ? stopDomain(row) : playDomain(row);
      }
    });
  });
  document.addEventListener('click', () => stopAllDomains());

  if (domainRows.length) {
    const galleryObserver = new IntersectionObserver((entries) => {
      entries.forEach((entry) => { if (!entry.isIntersecting) stopDomain(entry.target); });
    }, { threshold: 0.05 });
    domainRows.forEach((row) => galleryObserver.observe(row));
  }

  const viewportVideos = document.querySelectorAll('[data-viewport-video]');
  if (viewportVideos.length) {
    const videoObserver = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        const video = entry.target;
        if (entry.isIntersecting) {
          video.currentTime = 0;
          video.play().catch(() => {});
        } else {
          video.pause();
        }
      });
    }, { threshold: 0.45 });
    viewportVideos.forEach((video) => videoObserver.observe(video));
  }
});
