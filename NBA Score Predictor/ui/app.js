async function makePrediction() {
    const homeTeam = document.getElementById('hometeam').value;
    const awayTeam = document.getElementById('awayteam').value;
    const checkBoxes = document.querySelectorAll('.checkboxes input[type=checkbox]:checked');
    const checkBoxValue = Array.from(checkBoxes).map(checkbox => checkbox.value);

    const input = {
        homeTeam: homeTeam,
        awayTeam: awayTeam,
        playing: checkBoxValue
    };

    if (homeTeam == awayTeam) {
        alert('Teams selected cannot be the same!')
        return;
    }
    
    try {
        const response = await fetch('http://127.0.0.1:5000/nba_predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body:JSON.stringify(input)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP Error! Status: ${response.status}`);
        }

        const result = await response.json();

        document.getElementById('prediction').textContent = result[0]['prediction'];
        document.getElementById('probability').textContent = result[0]['probability'];
    } catch (error) {
        alert('Failed to get prediction! The NBA server might be down, please try again later');
    }
}