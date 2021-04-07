if [ -d "TruckDataset_TD" ] 
then
    echo "Directory TruckDataset_TD exists." 
else
    echo "Error: Directory TruckDataset_TD does not exists."
    wget https://github.com/trangdao909/TruckDetective/raw/main/TruckDataset_TD.zip
    unzip TruckDataset_TD.zip
fi
