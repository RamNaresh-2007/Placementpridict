let currentStudentPage = 1;
let totalStudentPages = 1;
let currentFilter = 'all';
let allCachedPlots = [];
let skillsRadarChartInstance = null;

document.addEventListener("DOMContentLoaded", () => {
    fetchDashboardData();
    fetchPCAData();
    fetchAnomalyData();
    fetchLeakageData();
    setupLightboxModal();
    setupDashboardPlotsExplorer();
    setupTabs();
    
    // Students logic
    fetchStudents(1, currentFilter);
    
    const prevBtn = document.getElementById('prev-page');
    if (prevBtn) {
        prevBtn.addEventListener('click', () => {
            if (currentStudentPage > 1) fetchStudents(currentStudentPage - 1, currentFilter);
        });
    }
    
    const nextBtn = document.getElementById('next-page');
    if (nextBtn) {
        nextBtn.addEventListener('click', () => {
            if (currentStudentPage < totalStudentPages) fetchStudents(currentStudentPage + 1, currentFilter);
        });
    }

    const dataFilter = document.getElementById('data-filter');
    if (dataFilter) {
        dataFilter.addEventListener('change', (e) => {
            currentFilter = e.target.value;
            fetchStudents(1, currentFilter);
        });
    }

    // Gallery logic
    fetchPlotsGallery();

    // AI Predictor & Auth initializers
    initPredictor();
    initAuth();
});

function setupTabs() {
    const navItems = document.querySelectorAll('#nav-menu li');
    const viewSections = document.querySelectorAll('.view-section');

    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            navItems.forEach(n => n.classList.remove('active'));
            viewSections.forEach(v => v.style.display = 'none');
            
            item.classList.add('active');
            const targetId = item.getAttribute('data-target');
            const targetSection = document.getElementById(targetId);
            if (targetSection) {
                targetSection.style.display = 'block';
            }

            // If navigating to predictor and radar not yet rendered, trigger prediction
            if (targetId === 'predictor-view' && !skillsRadarChartInstance) {
                runPrediction();
            }
        });
    });
}

// -----------------------------------------------------------------------------
// 1. Dashboard Analytics Logic
// -----------------------------------------------------------------------------
async function fetchDashboardData() {
    try {
        const response = await fetch('/api/data');
        const data = await response.json();
        
        if (data.error) {
            console.error("Error fetching data:", data.error);
            return;
        }

        updateKPIs(data.kpis);
        renderCharts(data.charts);
    } catch (error) {
        console.error("Failed to fetch dashboard data:", error);
    }
}

function updateKPIs(kpis) {
    animateValue("kpi-total-students", 0, kpis.total_students, 1000);
    document.getElementById("kpi-placement-rate").innerText = `${kpis.placement_rate}%`;
    document.getElementById("kpi-avg-cgpa").innerText = kpis.avg_cgpa;
    document.getElementById("kpi-avg-salary").innerText = kpis.avg_salary;
    
    if (document.getElementById("kpi-missing-values")) {
        animateValue("kpi-missing-values", 0, kpis.missing_values || 0, 1000);
    }
    if (document.getElementById("kpi-cleaned-records")) {
        animateValue("kpi-cleaned-records", 0, kpis.cleaned_records || 0, 1000);
    }
}

function animateValue(id, start, end, duration) {
    const obj = document.getElementById(id);
    if (!obj) return;
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        obj.innerHTML = Math.floor(progress * (end - start) + start).toLocaleString();
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}

function renderCharts(charts) {
    if (typeof Chart === 'undefined' || !charts) {
        console.warn("Chart.js not loaded or chart data missing.");
        return;
    }
    Chart.defaults.color = '#475569';
    Chart.defaults.borderColor = '#e2e8f0';
    Chart.defaults.font.family = "'Outfit', 'Inter', sans-serif";

    // 1. CGPA Histogram
    const ctxCgpa = document.getElementById('cgpaChart')?.getContext('2d');
    if (ctxCgpa) {
        new Chart(ctxCgpa, {
            type: 'bar',
            data: {
                labels: charts.cgpa_hist.labels,
                datasets: [{
                    label: 'Number of Students',
                    data: charts.cgpa_hist.data,
                    backgroundColor: 'rgba(79, 70, 229, 0.65)',
                    borderColor: 'rgba(79, 70, 229, 1)',
                    borderWidth: 1,
                    borderRadius: 5
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: { y: { beginAtZero: true } }
            }
        });
    }

    // 2. Placement Pie Chart
    const ctxPie = document.getElementById('placementPieChart')?.getContext('2d');
    if (ctxPie) {
        new Chart(ctxPie, {
            type: 'doughnut',
            data: {
                labels: charts.placement_status.labels,
                datasets: [{
                    data: charts.placement_status.data,
                    backgroundColor: [
                        'rgba(16, 185, 129, 0.85)', // Placed (Green)
                        'rgba(239, 68, 68, 0.85)',   // Not Placed (Red)
                        'rgba(245, 158, 11, 0.85)'   // Other
                    ],
                    borderColor: '#ffffff',
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { position: 'bottom' } },
                cutout: '62%'
            }
        });
    }

    // 3. Stream Bar Chart
    const ctxStream = document.getElementById('streamChart')?.getContext('2d');
    if (ctxStream) {
        new Chart(ctxStream, {
            type: 'bar',
            data: {
                labels: charts.stream_data.labels,
                datasets: [{
                    label: 'Students',
                    data: charts.stream_data.data,
                    backgroundColor: 'rgba(14, 165, 233, 0.65)',
                    borderColor: 'rgba(14, 165, 233, 1)',
                    borderWidth: 1,
                    borderRadius: 5
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: { y: { beginAtZero: true } }
            }
        });
    }
}

// -----------------------------------------------------------------------------
// 2. AI Predictor & Career Coach Logic (NEW FEATURE)
// -----------------------------------------------------------------------------
function initPredictor() {
    const sliders = [
        { id: 'input-cgpa', valId: 'val-cgpa', suffix: '' },
        { id: 'input-coding', valId: 'val-coding', suffix: '%' },
        { id: 'input-aptitude', valId: 'val-aptitude', suffix: '%' },
        { id: 'input-mock', valId: 'val-mock', suffix: '%' },
        { id: 'input-attendance', valId: 'val-attendance', suffix: '%' }
    ];

    sliders.forEach(s => {
        const input = document.getElementById(s.id);
        const label = document.getElementById(s.valId);
        if (input && label) {
            input.addEventListener('input', () => {
                label.innerText = `${input.value}${s.suffix}`;
            });
        }
    });

    // Form submit
    const form = document.getElementById('ai-predict-form');
    if (form) {
        form.addEventListener('submit', (e) => {
            e.preventDefault();
            runPrediction();
        });
    }

    // Quick Demos
    document.querySelectorAll('.demo-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.getElementById('input-cgpa').value = btn.dataset.cgpa;
            document.getElementById('val-cgpa').innerText = btn.dataset.cgpa;

            document.getElementById('input-coding').value = btn.dataset.coding;
            document.getElementById('val-coding').innerText = `${btn.dataset.coding}%`;

            document.getElementById('input-aptitude').value = btn.dataset.aptitude;
            document.getElementById('val-aptitude').innerText = `${btn.dataset.aptitude}%`;

            document.getElementById('input-mock').value = btn.dataset.mock;
            document.getElementById('val-mock').innerText = `${btn.dataset.mock}%`;

            document.getElementById('input-attendance').value = btn.dataset.attendance;
            document.getElementById('val-attendance').innerText = `${btn.dataset.attendance}%`;

            document.getElementById('input-internships').value = btn.dataset.internships || 1;
            document.getElementById('input-projects').value = btn.dataset.projects || 2;

            runPrediction();
        });
    });

    // Run initial prediction silently on load
    runPrediction();
}

async function runPrediction() {
    try {
        const payload = {
            cgpa: parseFloat(document.getElementById('input-cgpa')?.value || 7.5),
            coding_test_score: parseFloat(document.getElementById('input-coding')?.value || 65),
            aptitude_test_score: parseFloat(document.getElementById('input-aptitude')?.value || 70),
            mock_interview_score: parseFloat(document.getElementById('input-mock')?.value || 65),
            attendance_percent: parseFloat(document.getElementById('input-attendance')?.value || 85),
            internships: parseInt(document.getElementById('input-internships')?.value || 1),
            projects: parseInt(document.getElementById('input-projects')?.value || 2),
            college_tier: document.getElementById('input-tier')?.value || 'Tier2',
            history_of_backlogs: document.getElementById('input-backlogs')?.value || 'No'
        };

        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await res.json();
        if (data.error) {
            console.error("Prediction failed:", data.error);
            return;
        }

        renderPredictionResults(data);
    } catch (err) {
        console.error("Error executing prediction:", err);
    }
}

function renderPredictionResults(data) {
    const probPct = data.placement_probability_pct;
    
    // 1. Radial Ring & Probability
    const probVal = document.getElementById('pred-prob-val');
    if (probVal) probVal.innerText = `${probPct.toFixed(1)}%`;

    const probRing = document.getElementById('prob-ring');
    const badge = document.getElementById('pred-status-badge');
    
    let ringColor = '#10b981'; // Green
    let statusText = '🎉 High Placement Chance';
    let statusClass = 'badge-success';

    if (probPct < 45) {
        ringColor = '#ef4444'; // Red
        statusText = '⚠️ High Risk Cohort';
        statusClass = 'badge-danger';
    } else if (probPct < 75) {
        ringColor = '#f59e0b'; // Orange
        statusText = '⚖️ Borderline Candidate';
        statusClass = 'badge-warning';
    }

    if (probRing) {
        probRing.style.background = `conic-gradient(${ringColor} ${probPct}%, #e2e8f0 ${probPct}%)`;
    }
    if (badge) {
        badge.className = `pred-status-badge ${statusClass}`;
        badge.innerText = statusText;
    }

    // 2. Salary Package & Persona Tag
    const salaryVal = document.getElementById('pred-salary-val');
    if (salaryVal) {
        salaryVal.innerText = data.estimated_salary_lpa ? `${data.estimated_salary_lpa} LPA` : 'N/A';
    }

    const personaTag = document.getElementById('pred-persona-tag');
    if (personaTag) {
        personaTag.innerText = `Matched: ${data.matched_persona}`;
    }

    // 3. Company Fit Progress Bars
    if (data.company_fit) {
        const sDream = data.company_fit.super_dream_readiness;
        const dream = data.company_fit.dream_readiness;
        const core = data.company_fit.core_readiness;

        document.getElementById('fit-super-dream').innerText = `${sDream}%`;
        document.getElementById('bar-super-dream').style.width = `${sDream}%`;

        document.getElementById('fit-dream').innerText = `${dream}%`;
        document.getElementById('bar-dream').style.width = `${dream}%`;

        document.getElementById('fit-core').innerText = `${core}%`;
        document.getElementById('bar-core').style.width = `${core}%`;
    }

    // 4. Actionable AI Tips
    const tipsList = document.getElementById('ai-tips-list');
    if (tipsList && data.actionable_tips) {
        tipsList.innerHTML = data.actionable_tips.map(tip => `<li>${tip}</li>`).join('');
    }

    // 5. Radar Chart
    renderRadarChart(data.student_metrics, data.benchmarks);
}

function renderRadarChart(student, benchmark) {
    if (typeof Chart === 'undefined') return;
    const ctxRadar = document.getElementById('skillsRadarChart')?.getContext('2d');
    if (!ctxRadar) return;

    const labels = ['CGPA (x10)', 'Coding Score', 'Aptitude Score', 'Mock Interview', 'Attendance'];
    const studentData = [
        student.cgpa * 10,
        student.coding_test_score,
        student.aptitude_test_score,
        student.mock_interview_score,
        student.attendance_percent
    ];
    const benchmarkData = [
        benchmark.cgpa * 10,
        benchmark.coding_test_score,
        benchmark.aptitude_test_score,
        benchmark.mock_interview_score,
        benchmark.attendance_percent
    ];

    if (skillsRadarChartInstance) {
        skillsRadarChartInstance.data.datasets[0].data = studentData;
        skillsRadarChartInstance.data.datasets[1].data = benchmarkData;
        skillsRadarChartInstance.update();
    } else {
        skillsRadarChartInstance = new Chart(ctxRadar, {
            type: 'radar',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Your Current Profile',
                        data: studentData,
                        backgroundColor: 'rgba(79, 70, 229, 0.25)',
                        borderColor: '#4f46e5',
                        borderWidth: 2,
                        pointBackgroundColor: '#4f46e5'
                    },
                    {
                        label: 'Placed Student Benchmark',
                        data: benchmarkData,
                        backgroundColor: 'rgba(16, 185, 129, 0.12)',
                        borderColor: '#10b981',
                        borderWidth: 2,
                        borderDash: [5, 5],
                        pointBackgroundColor: '#10b981'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 100,
                        ticks: { stepSize: 20 }
                    }
                },
                plugins: {
                    legend: { position: 'bottom' }
                }
            }
        });
    }
}

// -----------------------------------------------------------------------------
// 3. User Authentication & Profile Portal Logic (NEW FEATURE)
// -----------------------------------------------------------------------------
function initAuth() {
    const modal = document.getElementById('auth-modal');
    const openBtns = [document.getElementById('auth-btn-top'), document.getElementById('user-pill-btn')];
    const closeBtn = document.getElementById('close-modal-btn');

    openBtns.forEach(btn => {
        if (btn) {
            btn.addEventListener('click', () => {
                if (modal) modal.style.display = 'flex';
            });
        }
    });

    if (closeBtn) {
        closeBtn.addEventListener('click', () => {
            if (modal) modal.style.display = 'none';
        });
    }

    if (modal) {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) modal.style.display = 'none';
        });
    }

    // Modal Form Login
    const loginBtn = document.getElementById('modal-login-btn');
    if (loginBtn) {
        loginBtn.addEventListener('click', async () => {
            const sid = document.getElementById('modal-student-id')?.value.trim();
            const name = document.getElementById('modal-user-name')?.value.trim() || 'Student';
            const role = document.getElementById('modal-user-role')?.value || 'Student';

            await handleUserLogin(sid, name, role);
            if (modal) modal.style.display = 'none';
        });
    }

    // 1-Click Quick Demo Login buttons
    document.querySelectorAll('.quick-login-btn').forEach(btn => {
        btn.addEventListener('click', async () => {
            const sid = btn.dataset.id;
            const name = btn.dataset.name;
            const role = btn.dataset.role;

            await handleUserLogin(sid, name, role);
            if (modal) modal.style.display = 'none';
        });
    });

    // Restore existing session if present
    const savedUser = localStorage.getItem('placement_ai_user');
    if (savedUser) {
        try {
            const userObj = JSON.parse(savedUser);
            updateUserUI(userObj);
        } catch (e) {}
    }
}

async function handleUserLogin(studentId, name, role) {
    try {
        const res = await fetch('/api/auth/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ student_id: studentId, name: name, role: role })
        });
        const data = await res.json();
        if (data.status === 'success') {
            localStorage.setItem('placement_ai_user', JSON.stringify(data.user));
            updateUserUI(data.user);

            // If real student profile returned, auto-fill the predictor
            if (data.user.profile) {
                applyProfileToPredictor(data.user.profile);
            }
        }
    } catch (e) {
        console.error("Login request failed:", e);
    }
}

function updateUserUI(user) {
    const nameEl = document.getElementById('display-user-name');
    const roleEl = document.getElementById('display-user-role');
    const authBtnText = document.getElementById('auth-btn-text');

    if (nameEl) nameEl.innerText = user.name || 'Student';
    if (roleEl) roleEl.innerText = user.role || 'Student';
    if (authBtnText) authBtnText.innerText = user.name ? `Signed In: ${user.name.split(' ')[0]}` : 'Sign In';
}

function applyProfileToPredictor(p) {
    if (p.CGPA) {
        document.getElementById('input-cgpa').value = p.CGPA;
        document.getElementById('val-cgpa').innerText = p.CGPA;
    }
    if (p.CodingTestScore) {
        document.getElementById('input-coding').value = p.CodingTestScore;
        document.getElementById('val-coding').innerText = `${p.CodingTestScore}%`;
    }
    if (p.AptitudeTestScore) {
        document.getElementById('input-aptitude').value = p.AptitudeTestScore;
        document.getElementById('val-aptitude').innerText = `${p.AptitudeTestScore}%`;
    }
    if (p.MockInterviewScore) {
        document.getElementById('input-mock').value = p.MockInterviewScore;
        document.getElementById('val-mock').innerText = `${p.MockInterviewScore}%`;
    }
    if (p.AttendancePercent) {
        document.getElementById('input-attendance').value = p.AttendancePercent;
        document.getElementById('val-attendance').innerText = `${p.AttendancePercent}%`;
    }
    if (p.Internships !== undefined) {
        document.getElementById('input-internships').value = p.Internships;
    }
    if (p.Projects !== undefined) {
        document.getElementById('input-projects').value = p.Projects;
    }

    runPrediction();
}

// -----------------------------------------------------------------------------
// 4. Students Table Pagination & Filter Logic
// -----------------------------------------------------------------------------
async function fetchStudents(page, filter = 'all') {
    try {
        const limit = 50;
        const response = await fetch(`/api/students?page=${page}&limit=${limit}&filter=${filter}`);
        const result = await response.json();
        
        if (result.error) {
            console.error("Error fetching students:", result.error);
            return;
        }

        currentStudentPage = result.page;
        totalStudentPages = result.total_pages;

        const tbody = document.getElementById('students-tbody');
        if (!tbody) return;
        tbody.innerHTML = '';
        
        result.data.forEach(student => {
            const tr = document.createElement('tr');
            const salary = student['Salary Package'] ? `${parseFloat(student['Salary Package']).toFixed(2)} LPA` : 'N/A';
            const cgpa = student['CGPA'] ? student['CGPA'].toFixed(2) : 'N/A';

            tr.innerHTML = `
                <td>${student.StudentID || '-'}</td>
                <td>${student.Gender || '-'}</td>
                <td>${student.Stream || '-'}</td>
                <td>${cgpa}</td>
                <td><span style="color: ${student.PlacementStatus === 'Placed' ? '#10b981' : '#ef4444'}; font-weight: 600;">${student.PlacementStatus || '-'}</span></td>
                <td>${salary}</td>
            `;
            tbody.appendChild(tr);
        });

        const pageInfo = document.getElementById('page-info');
        if (pageInfo) pageInfo.innerText = `Page ${currentStudentPage} of ${totalStudentPages}`;
        
        const prev = document.getElementById('prev-page');
        const next = document.getElementById('next-page');
        if (prev) prev.disabled = (currentStudentPage === 1);
        if (next) next.disabled = (currentStudentPage === totalStudentPages);

    } catch (error) {
        console.error("Failed to fetch students:", error);
    }
}

// -----------------------------------------------------------------------------
// 5. Reports & Plots Gallery Logic
// -----------------------------------------------------------------------------
let galleryCategory = 'all';
let gallerySearchQuery = '';

async function fetchPlotsGallery() {
    try {
        const response = await fetch('/api/plots');
        const result = await response.json();
        
        if (result.error) {
            console.error("Error fetching plots:", result.error);
            return;
        }

        allCachedPlots = result.plots_meta || [];
        
        // Setup category pills in Reports view
        const pillsWrap = document.getElementById('gallery-category-pills');
        if (pillsWrap && !pillsWrap.dataset.listenerAttached) {
            pillsWrap.dataset.listenerAttached = 'true';
            const pills = pillsWrap.querySelectorAll('.cat-pill');
            pills.forEach(pill => {
                pill.addEventListener('click', () => {
                    pills.forEach(p => p.classList.remove('active'));
                    pill.classList.add('active');
                    galleryCategory = pill.getAttribute('data-cat') || 'all';
                    renderPlots();
                });
            });
        }
        
        const searchInput = document.getElementById('plot-search');
        if (searchInput && !searchInput.dataset.listenerAttached) {
            searchInput.dataset.listenerAttached = 'true';
            searchInput.addEventListener('input', (e) => {
                gallerySearchQuery = e.target.value.toLowerCase().trim();
                renderPlots();
            });
        }
        
        renderPlots();
    } catch (error) {
        console.error("Failed to fetch plots gallery:", error);
    }
}

function renderPlots() {
    const gallery = document.getElementById('plots-gallery');
    if (!gallery) return;
    gallery.innerHTML = '';
    
    let filtered = allCachedPlots.filter(item => {
        if (galleryCategory !== 'all' && item.category_slug !== galleryCategory) {
            return false;
        }
        if (gallerySearchQuery) {
            const matchTitle = (item.title || '').toLowerCase().includes(gallerySearchQuery);
            const matchFile = (item.filename || '').toLowerCase().includes(gallerySearchQuery);
            const matchCat = (item.category || '').toLowerCase().includes(gallerySearchQuery);
            if (!matchTitle && !matchFile && !matchCat) return false;
        }
        return true;
    });

    const badge = document.getElementById('gallery-total-badge');
    if (badge) {
        badge.innerHTML = `<i class="ph ph-sparkle"></i> ${filtered.length} of ${allCachedPlots.length} Plots`;
    }

    if (filtered.length === 0) {
        gallery.innerHTML = '<p style="color: var(--text-muted); grid-column: 1/-1; text-align: center; padding: 40px; font-size: 15px;">No matching plots found for this filter.</p>';
        return;
    }

    filtered.forEach(plot => {
        const item = document.createElement('div');
        item.className = 'gallery-item';
        item.style.cursor = 'pointer';
        
        const plotUrl = `/plots/${plot.filename}`;
        item.innerHTML = `
            <img src="${plotUrl}" alt="${plot.title}" loading="lazy">
            <div class="gallery-item-title" title="${plot.title}">${plot.title}</div>
        `;
        const img = item.querySelector('img');
        img.onerror = () => {
            img.style.display = 'none';
            const errDiv = document.createElement('div');
            errDiv.className = 'plot-img-error';
            errDiv.innerHTML = '<i class="ph ph-warning-circle"></i><span>Plot preview unavailable</span>';
            item.insertBefore(errDiv, item.querySelector('.gallery-item-title'));
        };
        item.addEventListener('click', () => {
            openPlotLightbox(plotUrl, plot.title, plot.category, plot.description);
        });
        gallery.appendChild(item);
    });
}

// -----------------------------------------------------------------------------
// 6. PCA (Principal Component Analysis) Dynamic Data Fetcher
// -----------------------------------------------------------------------------
async function fetchPCAData() {
    try {
        const response = await fetch('/api/pca');
        const data = await response.json();
        
        if (data.status === 'success') {
            const pc1El = document.getElementById('pca-pc1-val');
            const pc2El = document.getElementById('pca-pc2-val');
            const pc12El = document.getElementById('pca-pc1-pc2-val');
            const t80El = document.getElementById('pca-thresh-80');
            const t95El = document.getElementById('pca-thresh-95');

            if (pc1El && data.pc1_variance_pct) pc1El.innerText = `${data.pc1_variance_pct}%`;
            if (pc2El && data.pc2_variance_pct) pc2El.innerText = `${data.pc2_variance_pct}%`;
            if (pc12El && data.pc1_pc2_cumulative_pct) pc12El.innerText = `${data.pc1_pc2_cumulative_pct}%`;
            if (t80El && data.thresholds && data.thresholds["80%"]) t80El.innerText = `${data.thresholds["80%"]} Comps`;
            if (t95El && data.thresholds && data.thresholds["95%"]) t95El.innerText = `${data.thresholds["95%"]} Comps`;
        }
    } catch (err) {
        console.warn("Could not fetch PCA data:", err);
    }
}

// -----------------------------------------------------------------------------
// 6b. Anomaly & Outlier Intelligence Dynamic Data Fetcher (Session 27)
// -----------------------------------------------------------------------------
async function fetchAnomalyData() {
    try {
        const response = await fetch('/api/anomaly');
        const data = await response.json();
        
        if (data.status === 'success') {
            const eeRocEl = document.getElementById('anomaly-ee-roc');
            const eeRecallEl = document.getElementById('anomaly-ee-recall');
            const isoRocEl = document.getElementById('anomaly-iso-roc');
            const dbPrecEl = document.getElementById('anomaly-db-precision');
            const detectedEl = document.getElementById('anomaly-detected-count');

            if (eeRocEl && data.best_covariance_model) {
                eeRocEl.innerText = `${(data.best_covariance_model.roc_auc * 100).toFixed(1)}%`;
            }
            if (eeRecallEl && data.best_covariance_model) {
                eeRecallEl.innerText = `${data.best_covariance_model.recall_pct}%`;
            }
            if (isoRocEl && data.primary_model) {
                isoRocEl.innerText = `${(data.primary_model.full_cohort_roc_auc * 100).toFixed(1)}%`;
            }
            if (dbPrecEl && data.best_density_model) {
                dbPrecEl.innerText = `${data.best_density_model.precision_pct}%`;
            }
            if (detectedEl && data.primary_model) {
                detectedEl.innerText = data.primary_model.detected_anomalies.toLocaleString();
            }
        }
    } catch (err) {
        console.warn("Could not fetch Anomaly detection data:", err);
    }
}

// -----------------------------------------------------------------------------
// 6c. Data Leakage & Cross-Validation Architecture Fetcher (Session 28)
// -----------------------------------------------------------------------------
async function fetchLeakageData() {
    try {
        const response = await fetch('/api/leakage');
        const data = await response.json();
        
        if (data.status === 'success' || data.vault_lock) {
            const vaultAccEl = document.getElementById('leakage-vault-acc');
            const vaultAucEl = document.getElementById('leakage-vault-auc');
            const nestedF1El = document.getElementById('leakage-nested-f1');
            const optimismDeltaEl = document.getElementById('leakage-optimism-delta');
            const noiseIllusionEl = document.getElementById('leakage-noise-illusion');

            if (vaultAccEl && data.vault_lock && data.vault_lock.vault_test_accuracy !== undefined) {
                vaultAccEl.innerText = `${(data.vault_lock.vault_test_accuracy * 100).toFixed(1)}%`;
            }
            if (vaultAucEl && data.vault_lock && data.vault_lock.vault_test_roc_auc !== undefined) {
                vaultAucEl.innerText = `${(data.vault_lock.vault_test_roc_auc * 100).toFixed(1)}%`;
            }
            if (nestedF1El && data.nested_cv && data.nested_cv.nested_f1_mean !== undefined) {
                nestedF1El.innerText = `${(data.nested_cv.nested_f1_mean * 100).toFixed(1)}%`;
            }
            if (optimismDeltaEl && data.nested_cv && data.nested_cv.tuning_optimism_delta !== undefined) {
                optimismDeltaEl.innerText = `+${(data.nested_cv.tuning_optimism_delta * 100).toFixed(2)}%`;
            }
            if (noiseIllusionEl && data.noise_selection_experiment) {
                noiseIllusionEl.innerText = `${(data.noise_selection_experiment.leaky_accuracy * 100).toFixed(1)}% vs ${(data.noise_selection_experiment.clean_accuracy * 100).toFixed(1)}%`;
            }
        }
    } catch (err) {
        console.warn("Could not fetch Data Leakage data:", err);
    }
}

// -----------------------------------------------------------------------------
// 7. Lightbox Zoom Modal for High-Resolution Plot Inspection
// -----------------------------------------------------------------------------
function openPlotLightbox(url, title, category, description) {
    const modal = document.getElementById('plot-lightbox-modal');
    if (!modal) return;
    
    const img = document.getElementById('lightbox-img');
    const titleEl = document.getElementById('lightbox-title');
    const catEl = document.getElementById('lightbox-category');
    const descEl = document.getElementById('lightbox-description');
    const dlBtn = document.getElementById('lightbox-download');

    if (img) img.src = url;
    if (titleEl) titleEl.innerText = title || 'Diagnostic Plot';
    if (catEl) catEl.innerText = category || 'Analysis';
    if (descEl) descEl.innerText = description || 'High-resolution diagnostic plot visualization.';
    if (dlBtn) {
        dlBtn.href = url;
        dlBtn.setAttribute('download', url.split('/').pop() || 'diagnostic_plot.png');
    }
    
    modal.style.display = 'flex';
    document.body.style.overflow = 'hidden';
}

function closePlotLightbox() {
    const modal = document.getElementById('plot-lightbox-modal');
    if (modal) {
        modal.style.display = 'none';
        document.body.style.overflow = '';
    }
}

function setupLightboxModal() {
    const closeBtn = document.getElementById('lightbox-close');
    const backdrop = document.getElementById('lightbox-backdrop');
    
    if (closeBtn) closeBtn.addEventListener('click', closePlotLightbox);
    if (backdrop) backdrop.addEventListener('click', closePlotLightbox);
    
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closePlotLightbox();
    });

    // Event delegation for all zoom trigger buttons and clickable plot wraps
    document.addEventListener('click', (e) => {
        const zoomBtn = e.target.closest('.btn-icon-zoom');
        const imgWrap = e.target.closest('.plot-img-wrap');
        const trigger = zoomBtn || imgWrap;
        
        if (trigger) {
            e.preventDefault();
            e.stopPropagation();
            const plotUrl = trigger.getAttribute('data-plot') || trigger.querySelector('img')?.src;
            const title = trigger.getAttribute('data-title') || 'Diagnostic Plot';
            const category = trigger.getAttribute('data-category') || 'Model Analysis';
            const card = trigger.closest('.plot-showcase-card');
            const desc = card ? card.querySelector('.plot-card-header p')?.innerText : '';
            if (plotUrl) {
                openPlotLightbox(plotUrl, title, category, desc);
            }
        }
    });
}

// -----------------------------------------------------------------------------
// 8. Comprehensive Dashboard Plots Explorer (Filter Pills + Search)
// -----------------------------------------------------------------------------
let allDashboardPlotsMeta = [];
let currentCategory = 'all';
let currentSearchQuery = '';
let currentPlotLimit = 24;

function updateCategoryCounts(plots) {
    const counts = { all: plots.length };
    plots.forEach(p => {
        const s = p.category_slug || 'other_eda';
        counts[s] = (counts[s] || 0) + 1;
    });

    Object.keys(counts).forEach(k => {
        const el = document.getElementById(`count-${k}`);
        if (el) el.innerText = counts[k];
        const gel = document.getElementById(`gcount-${k}`);
        if (gel) gel.innerText = counts[k];
    });
}

async function setupDashboardPlotsExplorer() {
    const grid = document.getElementById('dashboard-plots-grid');
    if (!grid) return;

    try {
        const response = await fetch('/api/plots');
        const data = await response.json();
        
        if (data.plots_meta && data.plots_meta.length > 0) {
            allDashboardPlotsMeta = data.plots_meta;
        } else if (data.plots) {
            allDashboardPlotsMeta = data.plots.map(p => ({
                filename: p,
                title: p.replace('.png', '').replace(/_/g, ' ').replace(/\//g, ' - '),
                category: 'Diagnostic Plots',
                category_slug: 'all',
                description: 'Placement predictability exploratory diagnostic chart.'
            }));
        }

        updateCategoryCounts(allDashboardPlotsMeta);

        // Setup category filter pills
        const pillsWrap = document.getElementById('dashboard-category-pills');
        if (pillsWrap && !pillsWrap.dataset.listenerAttached) {
            pillsWrap.dataset.listenerAttached = 'true';
            const pills = pillsWrap.querySelectorAll('.cat-pill');
            pills.forEach(pill => {
                pill.addEventListener('click', () => {
                    pills.forEach(p => p.classList.remove('active'));
                    pill.classList.add('active');
                    currentCategory = pill.getAttribute('data-cat') || 'all';
                    currentPlotLimit = 24;
                    const limitSelect = document.getElementById('dashboard-plots-limit');
                    if (limitSelect) limitSelect.value = "24";
                    renderDashboardPlots();
                });
            });
        }

        // Setup display limit select
        const limitSelect = document.getElementById('dashboard-plots-limit');
        if (limitSelect && !limitSelect.dataset.listenerAttached) {
            limitSelect.dataset.listenerAttached = 'true';
            limitSelect.addEventListener('change', (e) => {
                const val = e.target.value;
                currentPlotLimit = val === 'all' ? 99999 : parseInt(val);
                renderDashboardPlots();
            });
        }

        // Setup view all button
        const viewAllBtn = document.getElementById('dashboard-view-all-btn');
        if (viewAllBtn && !viewAllBtn.dataset.listenerAttached) {
            viewAllBtn.dataset.listenerAttached = 'true';
            viewAllBtn.addEventListener('click', () => {
                currentPlotLimit = 99999;
                if (limitSelect) limitSelect.value = "all";
                renderDashboardPlots();
            });
        }

        // Setup search input
        const searchInput = document.getElementById('dashboard-plots-search');
        const clearBtn = document.getElementById('clear-plots-search');
        if (searchInput && !searchInput.dataset.listenerAttached) {
            searchInput.dataset.listenerAttached = 'true';
            searchInput.addEventListener('input', (e) => {
                currentSearchQuery = e.target.value.toLowerCase().trim();
                if (clearBtn) clearBtn.style.display = currentSearchQuery ? 'block' : 'none';
                renderDashboardPlots();
            });
        }

        if (clearBtn && !clearBtn.dataset.listenerAttached) {
            clearBtn.dataset.listenerAttached = 'true';
            clearBtn.addEventListener('click', () => {
                if (searchInput) searchInput.value = '';
                currentSearchQuery = '';
                clearBtn.style.display = 'none';
                renderDashboardPlots();
            });
        }

        // Setup load more button
        const loadMoreBtn = document.getElementById('dashboard-load-more-btn');
        if (loadMoreBtn && !loadMoreBtn.dataset.listenerAttached) {
            loadMoreBtn.dataset.listenerAttached = 'true';
            loadMoreBtn.addEventListener('click', () => {
                currentPlotLimit += 24;
                renderDashboardPlots();
            });
        }

        renderDashboardPlots();
    } catch (err) {
        console.error("Failed to setup dashboard plots explorer:", err);
    }
}

function renderDashboardPlots() {
    const grid = document.getElementById('dashboard-plots-grid');
    if (!grid) return;

    let filtered = allDashboardPlotsMeta.filter(item => {
        // Filter by category slug
        if (currentCategory !== 'all' && item.category_slug !== currentCategory) {
            return false;
        }

        // Filter by search query
        if (currentSearchQuery) {
            const matchTitle = (item.title || '').toLowerCase().includes(currentSearchQuery);
            const matchFile = (item.filename || '').toLowerCase().includes(currentSearchQuery);
            const matchCat = (item.category || '').toLowerCase().includes(currentSearchQuery);
            if (!matchTitle && !matchFile && !matchCat) return false;
        }

        return true;
    });

    const totalMatching = filtered.length;
    const showing = filtered.slice(0, currentPlotLimit);

    grid.innerHTML = '';
    if (showing.length === 0) {
        grid.innerHTML = '<p style="color: var(--text-muted); grid-column: 1/-1; text-align: center; padding: 48px; font-size: 15px;">No matching diagnostic plots found.</p>';
    } else {
        showing.forEach(plot => {
            const card = document.createElement('div');
            card.className = 'master-plot-card';
            const plotUrl = `/plots/${plot.filename}`;
            card.innerHTML = `
                <img src="${plotUrl}" alt="${plot.title}" class="plot-thumb" loading="lazy">
                <div class="master-plot-card-body">
                    <span class="master-plot-cat">${plot.category || 'Diagnostic Plot'}</span>
                    <h5 class="master-plot-title" title="${plot.title}">${plot.title}</h5>
                </div>
            `;
            const img = card.querySelector('img');
            img.onerror = () => {
                img.style.display = 'none';
                const errDiv = document.createElement('div');
                errDiv.className = 'plot-img-error';
                errDiv.innerHTML = '<i class="ph ph-warning-circle"></i><span>Plot preview unavailable</span>';
                card.insertBefore(errDiv, card.querySelector('.master-plot-card-body'));
            };
            card.addEventListener('click', () => {
                openPlotLightbox(plotUrl, plot.title, plot.category, plot.description);
            });
            grid.appendChild(card);
        });
    }

    // Update counter and showing text
    const counterBadge = document.getElementById('dashboard-plots-counter');
    if (counterBadge) {
        counterBadge.innerHTML = `<i class="ph ph-sparkle"></i> ${allDashboardPlotsMeta.length} Diagnostic Plots`;
    }
    const showingText = document.getElementById('dashboard-showing-text');
    if (showingText) {
        showingText.innerText = `Showing ${Math.min(showing.length, totalMatching)} of ${totalMatching} plots`;
    }
    const loadMoreBtn = document.getElementById('dashboard-load-more-btn');
    const viewAllBtn = document.getElementById('dashboard-view-all-btn');
    const loadMoreWrap = document.getElementById('dashboard-load-more-wrap');
    if (loadMoreWrap) {
        loadMoreWrap.style.display = totalMatching === 0 ? 'none' : 'flex';
    }
    if (loadMoreBtn) {
        loadMoreBtn.style.display = totalMatching > showing.length ? 'inline-flex' : 'none';
    }
    if (viewAllBtn) {
        viewAllBtn.style.display = totalMatching > showing.length ? 'inline-flex' : 'none';
    }
}
