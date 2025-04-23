echo "running ${BASH_VERSION}"
SCRIPT_DIR=$(cd $(dirname $0); pwd)
PROJECT_DIR=$(cd $SCRIPT_DIR/..; pwd)
W=$(cd $PROJECT_DIR/..; pwd)
VAIP_DIR=$W/vaip

if [ ! -d $VAIP_DIR ]; then
    git clone git@gitenterprise.xilinx.com:VitisAI/vaip.git $VAIP_DIR
fi
echo "sync vaip"
(cd $VAIP_DIR; git fetch --all)
(cd $PROJECT_DIR; git fetch --all;)

old_commit_id=$(echo -n $(cd $VAIP_DIR; git show origin/cp_dev:cmake/deps.txt | grep -i 'MorphiZen;' | awk -F';' '{print $3}'))
echo "old commit id = $old_commit_id"

new_commit_id=$(cd $W/MorphiZen; git rev-parse origin/main)
echo "new commit id = $new_commit_id"

if [ x"$old_commit_id" == x"$new_commit_id" ]; then
    echo "no update"
    exit 0
fi

cd $VAIP_DIR;

branch_name="br_update_morphizen_from_${old_commit_id:0:8}"
echo "branch_name = $branch_name"
if  git rev-parse --verify "$branch_name" >/dev/null 2>&1; then
    echo "Branch $branch_name exists."
else
    git branch $branch_name
    git reset --hard origin/cp_dev
fi
git checkout --force $branch_name
git reset --hard origin/cp_dev
git show origin/cp_dev:cmake/deps.txt |
    sed "s/morphizen;\\(.*\\);\\(.*\\)/morphizen;\\1;$new_commit_id/g" > cmake/deps.txt;
git add cmake/deps.txt
title="update morphizen from ${old_commit_id:0:8} to ${new_commit_id:0:8}"
body=$(
    cd ../MorphiZen; echo "ChangeLog";
    env PAGER=cat git log --date=short --reverse --pretty=format:"- %h %s (by %an @ %ad)" $old_commit_id..$new_commit_id --date-order |
        sed 's:#\([0-9]*\):VitisAI/MorphiZen#\1:g'
    )
msg=$(echo $title
      echo
      echo $body)

git commit -am "$msg"
git push --force -u fork $branch_name

title="update vaip from ${old_commit_id:0:8} to ${new_commit_id:0:8}"
whoami=$(gh --hostname gitenterprise.xilinx.com api  user --jq .login)
gh pr create --base cp_dev --head $whoami:$branch_name --title "$title" --body "$body" || true
pr_number=$(gh pr list --head $branch_name --json number,title --jq '.[0].number')
if [ -z $pr_number ]; then
    echo "cannot find pr number"
    set -x
    gh pr list --head $branch_name --json number,title --jq '.[0].number'
    exit 1
fi
echo "pr_number = $pr_number"
gh pr edit $pr_number --title "$title" --body "$body"
