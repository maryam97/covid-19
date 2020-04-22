# base imports
import csv
import json

# self imports
import debug

class handler_csv:
    def __init__(self):
        debug.debug_print("CSV Handler is up", 1)

    # Get 2 attribute and their data types, then compare to each ohter
    def _isEqual(self, firstAttribute, secondAttribute, dataType):
        if dataType == 'int':
            try:
                return int(firstAttribute, 10) == int(secondAttribute, 10)  # ERROR-throw exception but works fine... idk
            except:
                debug.debug_print("ERROR, comparing: %s & %s" % (firstAttribute, secondAttribute), 2)
        else:
            return firstAttribute == secondAttribute

    # Simple function for testing encoding of file, no need at all
    def print_singleColumnCsvFile(self, csvFilename, columnName):
        csvData = []
        with open(csvFilename) as csvFile:
            csvDriver = csv.DictReader(csvFile)
            for row in csvDriver:
                csvData.append(row)

        debug.debug_print("CSV File Data: %s" % (csvFilename), 3)
        print(columnName)
        for row in csvData:
            print(row[columnName])

    # Function that merge two CSV files on 'commonColumn'
    def merge_csvFiles_byColumn(self, csvFilename1, csvFilename2, destinationFilename, commonColumnNameOnFile1, commonColumnNameOnFile2, commonColumnDataType='string'):
        # Get first csvFile data
        csvData1 = []
        csvFile1Fieldnames = []
        with open(csvFilename1) as csvFile:
            csvDriver = csv.DictReader(csvFile)
            csvFile1Fieldnames = csvDriver.fieldnames
            for row in csvDriver:
                csvData1.append(row)

        # Get secondary csvFile data
        csvData2 = []
        csvFile2Fieldnames = []
        with open(csvFilename2) as csvFile:
            csvDriver = csv.DictReader(csvFile)
            csvFile2Fieldnames = csvDriver.fieldnames
            for row in csvDriver:
                csvData2.append(row)
        '''
        debug.debug_print("File1 Fieldnames", 2)
        print(csvFile1Fieldnames)
        debug.debug_seprate()
        debug.debug_print("File 2Fieldnames", 2)
        print(csvFile2Fieldnames)
        debug.debug_seprate()
        '''
        # Generate fina fieldnames after merge and remove duplication of 'commonColumn'
        mergedDataFieldnames = csvFile1Fieldnames + csvFile2Fieldnames
        mergedDataFieldnames.remove(commonColumnNameOnFile2)

        # Merge and save data 
        with open(destinationFilename, 'w') as destinationFile:
            csvDriver = csv.DictWriter(destinationFile, fieldnames=mergedDataFieldnames)
            csvDriver.writeheader()

            for row in csvData1:
                # Find correspondingRow on csvData2
                found = False
                for item in csvData2:
                    if self._isEqual(item.get(commonColumnNameOnFile2), row.get(commonColumnNameOnFile1), commonColumnDataType):
                        correspondingRow = item
                        found = True
                        break

                if found == False:
                    debug.debug_print("county not found:\t%s:%s" % (commonColumnNameOnFile1, row.get(commonColumnNameOnFile1)), 3)
                    continue

                # Remove duplication of commonColumn and update row
                del correspondingRow[commonColumnNameOnFile2]
                row.update(correspondingRow)
                # Add row to file
                csvDriver.writerow(row)
        
        debug.debug_print("SUCCESS: merge completed", 2)

    # Function that merge rows of two CSV files
    def merge_csvFiles_byRow(self, csvFilename1, csvFilename2, destinationFilename):
        # Get first csvFile data
        csvData1 = []
        csvFile1Fieldnames = []
        with open(csvFilename1) as csvFile:
            csvDriver = csv.DictReader(csvFile)
            csvFile1Fieldnames = csvDriver.fieldnames
            for row in csvDriver:
                csvData1.append(row)

        # Get secondary csvFile data
        csvData2 = []
        csvFile2Fieldnames = []
        with open(csvFilename2) as csvFile:
            csvDriver = csv.DictReader(csvFile)
            csvFile2Fieldnames = csvDriver.fieldnames
            for row in csvDriver:
                csvData2.append(row)

        if csvFile1Fieldnames != csvFile2Fieldnames:
            debug.debug_print("ERROR: mismatch in columns", 2)
            return

        with open(destinationFilename, 'w') as destinationFile:
            csvDriver = csv.DictWriter(destinationFile, fieldnames=csvFile1Fieldnames)
            csvDriver.writeheader()
            csvDriver.writerows(csvData1)
            csvDriver.writerows(csvData2)

        debug.debug_print("SUCCESS: merge completed", 2)

    # Load a CSV file, save interested fields known as 'fieldnames' to DestinationFilename, ignore others
    def simplify_csvFile(self, csvFilename, destinationFilename, fieldnames):
        # Get csvFile data
        csvData = []
        with open(csvFilename) as csvFile:
            csvDriver = csv.DictReader(csvFile)
            for row in csvDriver:
                csvData.append(row)

        with open(destinationFilename, 'w') as DestinationFile:
            csvDriver = csv.DictWriter(DestinationFile, fieldnames)
            csvDriver.writeheader()
            for row in csvData:
                csvDriver.writerow({k:row[k] for k in (fieldnames) if k in row})
        
        debug.debug_print("SUCCESS: simplify completed", 2)

class handler_json:
    data = []
    def __init__(self):
        debug.debug_print("JSON Handler is up", 1)

    # Function that transform a JSON file to CSV file. Only the fields that mentioned in 'fieldnames'
    def transform_jsonToCsv(self, jsonFilename, csvFilename, fieldnames):
        jsonData = []
        with open(jsonFilename) as jsonFile:
            jsonData = json.load(jsonFile)[1:]
        with open(csvFilename, 'w') as csvFile:
            csvDriver = csv.DictWriter(csvFile, fieldnames=fieldnames)
            csvDriver.writeheader()

            for row in jsonData:
                csvDriver.writerow({k:unicode(row[k]).encode('utf-8') for k in (fieldnames) if k in row})

        debug.debug_print("SUCCESS: transform completed", 2)

    def transform_jsonToCsv_hospitalBedData(self, jsonFilename, csvFilename):
        jsonMetaData = []
        fieldnames = ['county_fips', 'countyName', 'stateName', 'beds', 'unoccupiedBeds']
        with open(jsonFilename) as jsonFile:
            jsonMetaData = json.load(jsonFile)

        jsonCountiesData = jsonMetaData['objects']['counties']['geometries']
        with open(csvFilename, 'w') as csvFile:
            csvDriver = csv.DictWriter(csvFile, fieldnames=fieldnames)
            csvDriver.writeheader()
            for row in jsonCountiesData:
                singleCountyData = {fieldnames[k]:unicode(row['properties'][fieldnames[k]]).encode('utf-8') for k in range(1, len(fieldnames))}
                singleCountyData['county_fips'] = int(row['id'], 10)
                csvDriver.writerow(singleCountyData)
        
        debug.debug_print("SUCCESS: transform completed(hospitalBedData)", 2)

    def transform_jsonToCsv_socialDistancingData(self, jsonFilename, csvFilename):
        jsonMetaData = []
        countyFieldnames = ['stateFips', 'stateName', 'countyFips', 'countyName']
        dataFieldnames = ['date', 'totalGrade', 'visitationGrade', 'encountersGrade', 'travelDistanceGrade']
        with open(jsonFilename) as jsonFile:
            jsonMetaData = json.load(jsonFile)

        jsonCountiesData = jsonMetaData['hits']['hits']
        singleCountyData = {}

        with open(csvFilename, 'w') as csvFile:
            csvDriver = csv.DictWriter(csvFile, fieldnames=(countyFieldnames+dataFieldnames))
            csvDriver.writeheader()
            for county in jsonCountiesData:
                # set county data
                singleCountyData = {field:county['_source'][field] for field in countyFieldnames}
                # set Grade of county for each day, specified with 'date'
                for data in county['_source']['data']:
                    singleCountyData.update({field:data[field] for field in dataFieldnames})
                    csvDriver.writerow(singleCountyData)
        
        debug.debug_print("SUCCESS: transform completed(socialDistancingData)", 2)

if __name__ == "__main__":
    jsonHandler = handler_json()
    csvHandler = handler_csv()
    
    # jsonHandler.transform_jsonToCsv('confirmanddeath.json', 'temp.csv', ['popul', 'county'])
    # csvHandler.merge e_csvFiles_byColumn('temp.csv', 'simpleCountyData.csv', 'merged.csv', 'county', 'CTYNAME')
    # csvHandler.simplify_csvFile('csvFiles/latitude.csv', 'csvFiles/simple-lat.csv', ['county_fips', 'lat'])
    # csvHandler.merge_csvFiles_byColumn('csvFiles/data.csv', 'csvFiles/simple-lat.csv', 'csvFiles/fixed-data.csv', 'county FIPS code', 'county_fips')
    # csvHandler.print_singleColumnCsvFile('csvFiles/simple-lat.csv', 'county_fips')
    # jsonHandler.transform_jsonToCsv_hospitalBedData('hospital-beds.json', 'csvFiles/hospital-beds.csv')
    # csvHandler.simplify_csvFile('csvFiles/hospital-beds.csv', 'csvFiles/simple-hospital-beds.csv', ['county_fips', 'beds(per1000)', 'unoccupiedBeds(per1000)'])
    # csvHandler.merge_csvFiles_byColumn('csvFiles/dataAndLat.csv', 'csvFiles/simple-hospital-beds.csv', 'csvFiles/fixed-data.csv', 'county FIPS code', 'county_fips')
    jsonHandler.transform_jsonToCsv_socialDistancingData('sd-state02.json', 'csvFiles/socialDistancing-s02.csv')
    # csvHandler.merge_csvFiles_byRow('csvFiles/socialDistancing-s01.csv', 'csvFiles/socialDistancing-s11.csv', 'csvFiles/socialDistancing.csv')