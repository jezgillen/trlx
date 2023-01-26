for (( ; ; ))
do
   du ray_results/ -h -d 0
   echo "Deleting checkpoints..."
   rm ray_results/*/*/checkpoint_* -r
   du ray_results/ -h -d 0
   sleep 30
done

