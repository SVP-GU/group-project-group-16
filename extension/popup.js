document.getElementById('checkButton').addEventListener('click', async () => {
    const text = document.getElementById('inputText').value;

    try {
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: text })
        });

        const data = await response.json();

        document.getElementById('result').innerText = 
            data.label === 'misinformation' 
            ? '⚠ Modellen varnar: Misinformation' 
            : '✅ Modellen säger: Sant';

    } catch (error) {
        document.getElementById('result').innerText = 'Fel vid anslutning till servern.';
        console.error('Fetch error:', error);
    }
});
