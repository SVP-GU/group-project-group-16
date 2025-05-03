async function analyzePageSentences() {
    console.log("🔍 Misinformation Checker analyserar meningar...");

    const pageText = document.body.innerText;
    const sentences = pageText.match(/[^.!?]+[.!?]+/g) || [];

    try {
        const response = await fetch('http://127.0.0.1:5000/batch_predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ texts: sentences })
        });

        const data = await response.json();
        const flaggedIndices = data.flagged || [];

        console.log(`⚠ Modellen hittade ${flaggedIndices.length} misstänkta meningar.`);

        let html = document.body.innerHTML;
        flaggedIndices.forEach(index => {
            const sentence = sentences[index].trim();
            const highlighted = `<span style="background-color: yellow; color: red; font-weight: bold;">${sentence}</span>`;
            const regex = new RegExp(sentence.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g');
            html = html.replace(regex, highlighted);
        });
        document.body.innerHTML = html;

        const banner = document.createElement('div');
        banner.id = 'misinfo-banner';
        banner.style.position = 'fixed';
        banner.style.top = '0';
        banner.style.left = '0';
        banner.style.width = '100%';
        banner.style.padding = '15px';
        banner.style.textAlign = 'center';
        banner.style.fontSize = '18px';
        banner.style.fontWeight = 'bold';
        banner.style.color = 'white';
        banner.style.backgroundColor = flaggedIndices.length > 0 ? '#e74c3c' : '#2ecc71';
        banner.style.zIndex = '9999';
        banner.innerText = flaggedIndices.length > 0
            ? `⚠ Modellen hittade ${flaggedIndices.length} misstänkta meningar!`
            : '✅ Ingen uppenbar misinformation hittad.';

        const closeBtn = document.createElement('span');
        closeBtn.innerText = ' ✖';
        closeBtn.style.marginLeft = '15px';
        closeBtn.style.cursor = 'pointer';
        closeBtn.onclick = () => banner.remove();
        banner.appendChild(closeBtn);

        document.body.prepend(banner);

    } catch (error) {
        console.error('Fel vid batch-koll:', error);
    }
}

analyzePageSentences();
