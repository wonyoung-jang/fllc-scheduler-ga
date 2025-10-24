const startButton = document.getElementById('startButton');
const statusContainer = document.getElementById('status-container');
const resultsContainer = document.getElementById('results-container');
const runIdSpan = document.getElementById('runId');
const statusSpan = document.getElementById('status');
const resultsList = document.getElementById('results-list');

let currentRunId = null;
let statusInterval = null;

// Function to start a new run
async function startRun() {
    startButton.disabled = true;
    startButton.textContent = 'Starting...';
    statusContainer.classList.remove('hidden');
    resultsContainer.classList.add('hidden');
    resultsList.innerHTML = '';
    statusSpan.textContent = 'Initializing...';

    try {
        // Fetch the default config to post
        const configResponse = await fetch('/config/default');
        const config = await configResponse.json();

        // Post the config to start a run
        const scheduleResponse = await fetch('/schedule', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        const data = await scheduleResponse.json();

        currentRunId = data.run_id;
        runIdSpan.textContent = currentRunId;
        startButton.textContent = 'Run in Progress...';

        // Start polling for status
        statusInterval = setInterval(checkStatus, 3000); // Check every 3 seconds
    } catch (error) {
        console.error('Error starting run:', error);
        statusSpan.textContent = 'Error starting run.';
        startButton.disabled = false;
        startButton.textContent = 'Start New Schedule Run';
    }
}

// Function to poll the status endpoint
async function checkStatus() {
    if (!currentRunId) return;

    try {
        const response = await fetch(`/status/${currentRunId}`);
        const data = await response.json();
        statusSpan.textContent = data.status;

        if (data.status === 'COMPLETED') {
            clearInterval(statusInterval);
            startButton.disabled = false;
            startButton.textContent = 'Start New Schedule Run';
            await fetchResults();
        } else if (data.status === 'FAILED') {
            clearInterval(statusInterval);
            startButton.disabled = false;
            startButton.textContent = 'Start New Schedule Run';
        }
    } catch (error) {
        console.error('Error checking status:', error);
        statusSpan.textContent = 'Error checking status.';
        clearInterval(statusInterval);
    }
}

// Function to fetch and display results
async function fetchResults() {
    try {
        const response = await fetch(`/results/${currentRunId}`);
        const data = await response.json();

        resultsContainer.classList.remove('hidden');
        data.files.forEach(file => {
            const li = document.createElement('li');
            const a = document.createElement('a');
            a.href = `/results/${currentRunId}/${file}`;
            a.textContent = file;
            a.target = '_blank'; // Open in new tab
            li.appendChild(a);
            resultsList.appendChild(li);
        });
    } catch (error) {
        console.error('Error fetching results:', error);
    }
}

startButton.addEventListener('click', startRun);
