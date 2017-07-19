#!/bin/bash
# Use > 1 to consume two arguments per pass in the loop (e.g. each argument has 
# a corresponding value to go with it). Use >0 to consume one or more arguments
# per pass in the loop (eg. some arguments don't have a corresponding value to 
# go with it such as in the --default example).
# note: if this is set to > 0 the /etc/hosts part is not recognized (may be a 
# bug)

while [[ $# > 1 ]]
do
    key="$1"

    case $key in
        -d|--directory)
        BASE_DIR="$2"
        shift # past argument
        ;;
        
        -n|--num-seeds)
        NUM_SEEDS="$2"
        shift # past argument
        ;;
        *)
        ;;
    esac
    shift # past argument or value
done

#if [[ -n $1 ]]; then
#    echo "Last line of file specified as non-opt/last argument:"
#    tail -1 $1
#fi

SEED=1
while [ "$SEED" -le "$NUM_SEEDS" ]
do 
    echo "Creating seed $SEED folder..."
    cp -r $BASE_DIR/seed-0 $BASE_DIR/seed-$SEED 
    ifname=$BASE_DIR/seed-$SEED/models/all-mod.cfg 
    ofname=$ifname.out; 
    sed -e s/seed:\\\ \\\[0\\\]/seed:\\\ \\\[$SEED\\\]/g $ifname > $ofname 
    mv $ofname $ifname

    SEED=$(($SEED + 1))
done
