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
        -c|--code-dir)
        CODE_DIR="$2"
        shift # past argument
        ;;

        -e|--expt-dir)
        EXPT_DIR="$2"
        shift # past argument
        ;;
        
        -n|--expt-name)
        EXPT_NAME="$2"
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

echo "Preparing job for $EXPT_NAME"
sed -e s#EXPT_NAME#$EXPT_NAME#g \
    -e s#CODE_DIR#$CODE_DIR#g \
    -e s#EXPT_DIR#$EXPT_DIR#g \
    template.job > run-this.job
    
qsub run-this.job
mv run-this.job $EXPT_DIR
