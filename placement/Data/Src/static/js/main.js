let currentStudentPage = 1;
let totalStudentPages = 1;
let currentFilter = 'all';

document.addEventListener("DOMContentLoaded", () => {
    fetchDashboardData();
    setupTabs();
    
    // Students logic
    fetchStudents(1, currentFilter);
    
    document.getElementById('prev-page').addEventListener('click', () => {
        if (currentStudentPage > 1) fetchStudents(currentStudentPage - 1, currentFilter);
    });
    document.getElementById('next-page').addEventListener('click', () => {
        if (currentStudentPage < totalStudentPages) fetchStudents(currentStudentPage + 1, currentFilter);
    });

    document.getElementById('data-filter').addEventListener('change', (e) => {
        currentFilter = e.target.value;
        fetchStudents(1, currentFilter);
    });

    // Gallery logic
    fetchPlotsGallery();
});

function setupTabs() {
    const navItems = document.querySelectorAll('#nav-menu li');
    const viewSections = document.querySelectorAll('.view-section');

    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            // Remove active from all
            navItems.forEach(n => n.classList.remove('active'));
            viewSections.forEach(v => v.style.display = 'none');
            
            // Add active to clicked
            item.classList.add('active');
            const targetId = item.getAttribute('data-target');
            document.getElementById(targetId).style.display = 'block';
        });
    });
}

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
        tbody.innerHTML = ''; // clear existing
        
        result.data.forEach(student => {
            const tr = document.createElement('tr');
            
            // Handle potentially missing values gently
            const salary = student['Salary Package'] ? `${parseFloat(student['Salary Package']).toFixed(2)} LPA` : 'N/A';
            const cgpa = student['CGPA'] ? student['CGPA'].toFixed(2) : 'N/A';

            tr.innerHTML = `
                <td>${student.StudentID || '-'}</td>
                <td>${student.Gender || '-'}</td>
                <td>${student.Stream || '-'}</td>
                <td>${cgpa}</td>
                <td><span style="color: ${student.PlacementStatus === 'Placed' ? '#2ea043' : '#f85149'}">${student.PlacementStatus || '-'}</span></td>
                <td>${salary}</td>
            `;
            tbody.appendChild(tr);
        });

        document.getElementById('page-info').innerText = `Page ${currentStudentPage} of ${totalStudentPages}`;
        
        document.getElementById('prev-page').disabled = (currentStudentPage === 1);
        document.getElementById('next-page').disabled = (currentStudentPage === totalStudentPages);

    } catch (error) {
        console.error("Failed to fetch students:", error);
    }
}

async function fetchPlotsGallery() {
    try {
        const response = await fetch('/api/plots');
        const result = await response.json();
        
        if (result.error) {
            console.error("Error fetching plots:", result.error);
            return;
        }

        const gallery = document.getElementById('plots-gallery');
        gallery.innerHTML = '';
        
        result.plots.forEach(plot => {
            const item = document.createElement('div');
            item.className = 'gallery-item';
            
            // Remove .png and underscores for the title
            const title = plot.replace('.png', '').replace(/_/g, ' ');
            
            item.innerHTML = `
                <img src="/plots/${plot}" alt="${title}" loading="lazy">
                <div class="gallery-item-title">${title}</div>
            `;
            gallery.appendChild(item);
        });
    } catch (error) {
        console.error("Failed to fetch plots gallery:", error);
    }
}

function updateKPIs(kpis) {
    // Animate numbers for a premium feel
    animateValue("kpi-total-students", 0, kpis.total_students, 1000);
    document.getElementById("kpi-placement-rate").innerText = `${kpis.placement_rate}%`;
    document.getElementById("kpi-avg-cgpa").innerText = kpis.avg_cgpa;
    document.getElementById("kpi-avg-salary").innerText = kpis.avg_salary;
    
    // New KPIs
    if (document.getElementById("kpi-missing-values")) {
        animateValue("kpi-missing-values", 0, kpis.missing_values || 0, 1000);
    }
    if (document.getElementById("kpi-cleaned-records")) {
        animateValue("kpi-cleaned-records", 0, kpis.cleaned_records || 0, 1000);
    }
}

function animateValue(id, start, end, duration) {
    const obj = document.getElementById(id);
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
    // Global Chart Defaults for light theme
    Chart.defaults.color = '#475569';
    Chart.defaults.borderColor = '#e2e8f0';
    Chart.defaults.font.family = "'Outfit', 'Inter', sans-serif";

    // 1. CGPA Histogram
    const ctxCgpa = document.getElementById('cgpaChart').getContext('2d');
    new Chart(ctxCgpa, {
        type: 'bar',
        data: {
            labels: charts.cgpa_hist.labels,
            datasets: [{
                label: 'Number of Students',
                data: charts.cgpa_hist.data,
                backgroundColor: 'rgba(88, 166, 255, 0.6)',
                borderColor: 'rgba(88, 166, 255, 1)',
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: { beginAtZero: true }
            }
        }
    });

    // 2. Placement Pie Chart
    const ctxPie = document.getElementById('placementPieChart').getContext('2d');
    new Chart(ctxPie, {
        type: 'doughnut',
        data: {
            labels: charts.placement_status.labels,
            datasets: [{
                data: charts.placement_status.data,
                backgroundColor: [
                    'rgba(16, 185, 129, 0.8)', // Placed (Green)
                    'rgba(239, 68, 68, 0.8)', // Not Placed (Red)
                    'rgba(245, 158, 11, 0.8)' // Other
                ],
                borderColor: '#ffffff',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'bottom' }
            },
            cutout: '60%'
        }
    });

    // 3. Stream Bar Chart
    const ctxStream = document.getElementById('streamChart').getContext('2d');
    new Chart(ctxStream, {
        type: 'bar',
        data: {
            labels: charts.stream_data.labels,
            datasets: [{
                label: 'Students',
                data: charts.stream_data.data,
                backgroundColor: 'rgba(163, 113, 247, 0.6)',
                borderColor: 'rgba(163, 113, 247, 1)',
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: { beginAtZero: true }
            }
        }
    });
}
