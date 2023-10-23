const fetch = require('node-fetch');
const fs = require('fs');
const readline = require('readline');

const handleQuerySubmit = async (csvFile, columnName, apiEndpoint) => {
    const formData = new FormData();
    formData.append('csvFile', csvFile);
    formData.append('columnName', columnName);

    try {
        const response = await fetch(apiEndpoint, {
            method: 'POST',
            body: formData,
        });
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error:', error);
        throw error;
    }
};

const runScript = async () => {
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
    });

    let apiEndpoint = 'https://your-api-endpoint.com/topics'; // Replace with your actual API endpoint

    rl.question('Enter the path to your CSV file: ', (csvFilePath) => {
        rl.question('Enter the name of the column to choose: ', (columnName) => {
            const csvFile = fs.createReadStream(csvFilePath);
            console.log('Uploading CSV file and column...');

            handleQuerySubmit(csvFile, columnName, apiEndpoint)
                .then((data) => {
                    console.log('API Response:', data);
                })
                .catch((error) => {
                    console.error('An error occurred:', error);
                })
                .finally(() => {
                    rl.close();
                });
        });
    });
};

// Run the script
runScript();
